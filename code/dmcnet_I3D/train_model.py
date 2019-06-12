"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import logging

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from data import iterator_factory
from train import metric
from train.model import model
from train.lr_scheduler import MultiFactorScheduler


def train_model(net_name,  sym_net, model_prefix, dataset, input_conf,
                modality = 'rgb', split = 1, clip_length=16, train_frame_interval=2, val_frame_interval=2,
                resume_epoch=-1, batch_size=4, save_frequency=1,
                lr_base=0.01, lr_base2=0.01, lr_d = None, lr_factor=0.1, lr_steps=[400000, 800000],
                end_epoch=1000, distributed=False, 
                pretrained_3d=None, fine_tune=False,iter_size = 1, optim = 'sgd', accumulate = True, ds_factor = 16,
                epoch_thre = 1, score_dir =None, mv_minmaxnorm = False, mv_loadimg = False, detach = False,
                adv = 0, new_classifier = False,
                **kwargs):

    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.multiprocessing.set_sharing_strategy('file_system')
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    # data iterator
    iter_seed = torch.initial_seed() \
                + (torch.distributed.get_rank() * 10 if distributed else 100) \
                + max(0, resume_epoch) * 100

    train_iter, eval_iter = iterator_factory.creat(name=dataset,
                                                   batch_size=batch_size,
                                                   clip_length=clip_length,
                                                   train_interval=train_frame_interval,
                                                   val_interval=val_frame_interval,
                                                   mean=input_conf['mean'],
                                                   std=input_conf['std'],
                                                   seed=iter_seed,
                                                   modality = modality,
                                                   split = split,
                                                   net_name = net_name,
                                                   accumulate = accumulate,
                                                   ds_factor = ds_factor, mv_minmaxnorm = mv_minmaxnorm, mv_loadimg =mv_loadimg)
    #define an instance of class model
    net = model(net=sym_net,
                criterion=torch.nn.CrossEntropyLoss().cuda(),
                model_prefix=model_prefix,
                step_callback_freq=50,
                save_checkpoint_freq=save_frequency,
                opt_batch_size=batch_size, # optional
                criterion2 = torch.nn.MSELoss().cuda() if modality == 'flow+mp4' else None,
                criterion3 = torch.nn.CrossEntropyLoss().cuda() if adv > 0. else None,
                adv = adv,
                )
    net.net.cuda()
    print(torch.cuda.current_device(), torch.cuda.device_count())
    # config optimization
    param_base_layers = []
    param_new_layers = []
    name_base_layers = []
    params_gf = []
    params_d = []
    for name, param in net.net.named_parameters():
        if modality == 'flow+mp4':
            if name.startswith('gen_flow_model'):
                params_gf.append(param)
            elif name.startswith('discriminator'):
                params_d.append(param)
            else:
                if (name.startswith('conv3d_0c_1x1') or name.startswith('classifier')):
                    #if name.startswith('classifier'): 
                    param_new_layers.append(param)
                else:
                    param_base_layers.append(param)
                    name_base_layers.append(name)
            #else:
            #    #print(name)
            #    param_new_layers.append(param)
        else:
            if fine_tune:
                if name.startswith('classifier') or name.startswith('conv3d_0c_1x1'):
                #if name.startswith('classifier'): 
                    param_new_layers.append(param)
                else:
                    param_base_layers.append(param)
                    name_base_layers.append(name)
            else:
                param_new_layers.append(param)
    if modality == 'flow+mp4':       
        if fine_tune:
            lr_mul = 0.2
        else:
            lr_mul = 0.5
    else:
        lr_mul = 0.2
    #print(params_d)
    if name_base_layers:
        out = "[\'" + '\', \''.join(name_base_layers) + "\']"
        logging.info("Optimizer:: >> recuding the learning rate of {} params: {} by factor {}".format(len(name_base_layers),
                     out if len(out) < 300 else out[0:150] + " ... " + out[-150:], lr_mul))
    if net_name == 'I3D':
        weight_decay = 0.0001
    elif net_name == 'MFNet_3D':
        weight_decay = 0.0001
    logging.info("Train_Model:: weight_decay: `{}'".format(weight_decay))
    if distributed:
        net.net = torch.nn.parallel.DistributedDataParallel(net.net).cuda()
    else:
        net.net = torch.nn.DataParallel(net.net).cuda()
    
    if optim == 'adam':
        optimizer = torch.optim.Adam([{'params': param_base_layers, 'lr_mult': lr_mul},
                                 {'params': param_new_layers, 'lr_mult': 1.0}],
                                lr=lr_base,
                                weight_decay=weight_decay)
        optimizer_2 = torch.optim.Adam([{'params': param_base_layers, 'lr_mult': lr_mul},
                                 {'params': param_new_layers, 'lr_mult': 1.0}],
                                lr = lr_base2,
                                weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD([{'params': param_base_layers, 'lr_mult': lr_mul},
                                 {'params': param_new_layers, 'lr_mult': 1.0}],
                                lr=lr_base,
                                momentum=0.9,
                                weight_decay=weight_decay,
                                nesterov=True)
        optimizer_2 = torch.optim.SGD([{'params': param_base_layers, 'lr_mult': lr_mul},
                                 {'params': param_new_layers, 'lr_mult': 1.0}],
                                lr=lr_base2,
                                momentum=0.9,
                                weight_decay=weight_decay,
                                nesterov=True)
    if adv > 0.:
        optimizer_3 = torch.optim.Adam(
        params_d,
        lr=lr_base,
        weight_decay = weight_decay,
        eps=0.001)
    else:
        optimizer_3 = None
    if modality == 'flow+mp4':
        if optim == 'adam':
            optimizer_mse = torch.optim.Adam(
            params_gf,
            lr=lr_base,
            weight_decay=weight_decay,
            eps = 1e-08)
            optimizer_mse_2 = torch.optim.Adam(
            params_gf,
            lr = lr_base2,
            weight_decay=weight_decay,
            eps = 0.001)
        else:
            optimizer_mse = torch.optim.SGD(
            params_gf,
            lr=lr_base,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True)
            optimizer_mse_2 = torch.optim.SGD(
            params_gf,
            lr = lr_base2,
            momentum=0.9,                
            weight_decay=weight_decay,
            nesterov=True)
    else:
        optimizer_mse = None
        optimizer_mse_2 = None
    # load params from pretrained 3d network
    if pretrained_3d and not pretrained_3d == 'False':
        if resume_epoch < 0:
            assert os.path.exists(pretrained_3d), "cannot locate: `{}'".format(pretrained_3d)
            logging.info("Initializer:: loading model states from: `{}'".format(pretrained_3d))
            if net_name == 'I3D':
                checkpoint = torch.load(pretrained_3d)
                keys = list(checkpoint.keys())
                state_dict = {}
                for name in keys:
                    state_dict['module.' + name] = checkpoint[name]
                del checkpoint
                net.load_state(state_dict, strict=False)
                if new_classifier:
                    checkpoint = torch.load('./network/pretrained/model_flow.pth')
                    keys = list(checkpoint.keys())
                    state_dict = {}
                    for name in keys:
                        state_dict['module.' + name] = checkpoint[name]
                    del checkpoint
                    net.load_state(state_dict, strict=False)
            else:
                checkpoint = torch.load(pretrained_3d)
                net.load_state(checkpoint['state_dict'], strict=False)
        else:
            logging.info("Initializer:: skip loading model states from: `{}'"
                + ", since it's going to be overwrited by the resumed model".format(pretrained_3d))

    # resume training: model and optimizer
    if resume_epoch < 0:
        epoch_start = 0
        step_counter = 0
    else:
        net.load_checkpoint(epoch=resume_epoch, optimizer=optimizer, optimizer_mse = optimizer_mse)
        epoch_start = resume_epoch
        step_counter = epoch_start * train_iter.__len__()

    # set learning rate scheduler
    num_worker = dist.get_world_size() if torch.distributed._initialized else 1
    lr_scheduler = MultiFactorScheduler(base_lr=lr_base,
                                        steps=[int(x/(batch_size*num_worker)) for x in lr_steps],
                                        factor=lr_factor,
                                        step_counter=step_counter)
    if modality == 'flow+mp4':
        lr_scheduler2 = MultiFactorScheduler(base_lr= lr_base2,
                                        steps=[int(x/(batch_size*num_worker)) for x in lr_steps],
                                        factor=lr_factor,
                                        step_counter=step_counter)
        if lr_d == None:
            lr_scheduler3 = MultiFactorScheduler(base_lr= lr_d,
                                        steps=[int(x/(batch_size*num_worker)) for x in lr_steps],
                                        factor=lr_factor,
                                        step_counter=step_counter)
        else:
            print("_____________",lr_d)
            lr_scheduler3 = MultiFactorScheduler(base_lr= lr_d,
                                        steps=[int(x/(batch_size*num_worker)) for x in lr_steps],
                                        factor=lr_factor,
                                        step_counter=step_counter)
    else:
        lr_scheduler2 = None
        lr_scheduler3 = None
    # define evaluation metric
    metrics_D = None
    if modality == 'flow+mp4':
        metrics = metric.MetricList(metric.Loss(name="loss-ce"),
                                    metric.Loss(name="loss-mse"),
                                metric.Accuracy(name="top1", topk=1),
                                metric.Accuracy(name="top5", topk=5),)
        if adv > 0:
            metrics_D = metric.MetricList(metric.Loss(name="classi_D"),
                                        metric.Loss(name="adv_D"))          
            
    else:
        metrics = metric.MetricList(metric.Loss(name="loss-ce"),
                                metric.Accuracy(name="top1", topk=1),
                                metric.Accuracy(name="top5", topk=5),)
    # enable cudnn tune
    cudnn.benchmark = True
    net.fit(train_iter=train_iter,
            eval_iter=eval_iter,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            epoch_start=epoch_start,
            epoch_end=end_epoch,
            iter_size = iter_size,
            optimizer_mse = optimizer_mse,
            optimizer_2 = optimizer_2,
            optimizer_3 = optimizer_3,
            optimizer_mse_2 = optimizer_mse_2, lr_scheduler2 = lr_scheduler2, lr_scheduler3 = lr_scheduler3,
            metrics_D = metrics_D,
            epoch_thre = epoch_thre, score_dir = score_dir, detach = detach)
