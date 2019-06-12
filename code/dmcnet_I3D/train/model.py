"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import time
import socket
import logging

import torch
import numpy as np
from . import metric
from . import callback


"""
Static Model
"""
class static_model(object):

    def __init__(self,
                 net,
                 criterion=None,
                 model_prefix='',
                 criterion2 = None,
                 criterion3 = None,
                 **kwargs):
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        # init params
        self.net = net
        self.model_prefix = model_prefix
        self.criterion = criterion
        self.criterion2 = criterion2
        self.criterion3 = criterion3

    def load_state(self, state_dict, strict=False):
        if strict:
            self.net.load_state_dict(state_dict=state_dict)
        else:
            # customized partialy load function
            net_state_keys = list(self.net.state_dict().keys())
            if 'module.optimizer' in state_dict.keys():
                state_dict = state_dict['module.state_dict']
            for name, param in state_dict.items():
                #print(name)
                if name in self.net.state_dict().keys():
                    dst_param_shape = self.net.state_dict()[name].shape
                    #print(name,dst_param_shape,dst_param_shape)
                    if param.shape == dst_param_shape:
                        self.net.state_dict()[name].copy_(param.view(dst_param_shape))
                        net_state_keys.remove(name)
                    elif name == 'module.conv1.conv.weight' or 'module.conv3d.1a.7x7.conv3d.weight':
                        logging.warning("rgb model for flow", dst_param_shape[1])
                        self.net.state_dict()[name].copy_((torch.mean(param, dim = 1, keepdim = True)).expand([-1, dst_param_shape[1], -1, -1, -1]))
                        net_state_keys.remove(name)
            # indicating missed keys
            if net_state_keys:
                logging.warning(">> Failed to load: {}".format(net_state_keys))
                return False
        return True

    def get_checkpoint_path(self, epoch):
        assert self.model_prefix, "model_prefix undefined!"
        if torch.distributed._initialized:
            hostname = socket.gethostname()
            checkpoint_path = "{}_at-{}_ep-{:04d}.pth".format(self.model_prefix, hostname, epoch)
        else:
            checkpoint_path = "{}_ep-{:04d}.pth".format(self.model_prefix, epoch)
        return checkpoint_path

    def load_checkpoint(self, epoch, optimizer=None, optimizer_mse=None):

        load_path = self.get_checkpoint_path(epoch)
        assert os.path.exists(load_path), "Failed to load: {} (file not exist)".format(load_path)

        checkpoint = torch.load(load_path)

        all_params_matched = self.load_state(checkpoint['state_dict'], strict=False)

        if optimizer:
            if 'optimizer' in checkpoint.keys() and all_params_matched:
                #load optimizer
                optimizer.load_state_dict(checkpoint['optimizer'])
                logging.info("Model & Optimizer states are resumed from: `{}'".format(load_path))
            else:
                logging.warning(">> Failed to load optimizer state from: `{}'".format(load_path))
            if not optimizer_mse == None and 'optimizer_mse' in checkpoint.keys() and all_params_matched:
                optimizer_mse.load_state_dict(checkpoint['optimizer_mse'])
                logging.info("Optimizer MSE states are resumed from: `{}'".format(load_path))
            else:
                logging.warning(">> Failed to load optimizer MSE state from: `{}'".format(load_path))
        else:
            logging.info("Only model state resumed from: `{}'".format(load_path))

        if 'epoch' in checkpoint.keys():
            if checkpoint['epoch'] != epoch:
                logging.warning(">> Epoch information inconsistant: {} vs {}".format(checkpoint['epoch'], epoch))

    def save_checkpoint(self, epoch, optimizer_state=None, optimizer_mse_state = None):

        save_path = self.get_checkpoint_path(epoch)
        save_folder = os.path.dirname(save_path)

        if not os.path.exists(save_folder):
            logging.debug("mkdir {}".format(save_folder))
            os.makedirs(save_folder)

        if not optimizer_state:
            torch.save({'epoch': epoch,
                        'state_dict': self.net.state_dict()},
                        save_path)
            logging.info("Checkpoint (only model) saved to: {}".format(save_path))
        else:
            if optimizer_mse_state == None:
                torch.save({'epoch': epoch,
                            'state_dict': self.net.state_dict(),
                            'optimizer': optimizer_state},
                            save_path)
            else:
                #we have different optimizer for classifier and generator, 
                #which are named repectively optimizer and optimizer_mse
                torch.save({'epoch': epoch,
                            'state_dict': self.net.state_dict(),
                            'optimizer': optimizer_state,
                            'optimizer_mse': optimizer_mse_state,},
                            save_path)
            logging.info("Checkpoint (model & optimizer) saved to: {}".format(save_path))


    def forward(self, data, target, node = 'logit', detach = False, stage = None):
        """ typical forward function with:
            typical output and typical loss
        """
        data = data.float().cuda()
        target = target.cuda()
        if self.net.training:
            input_var = torch.autograd.Variable(data, requires_grad=False)
            target_var = torch.autograd.Variable(target, requires_grad=False)
            if self.criterion2 == None:
                output = self.net(input_var)
            else:
                output, flow = self.net(input_var[:, :5, :, :, :], node = 'flow+logit', detach = detach)
                if not stage == None:
                    # Adversarial ground truths
                    valid_var = torch.cat([target_var.clone().fill_(1)]*(flow.size(2)), 0).cuda()
                    valid_var.requires_grad = False
                    fake_var = torch.cat([target_var.clone().fill_(0)]*(flow.size(2)), 0).cuda()
                    fake_var.requires_grad = False
                    #manually set the height and width of generated DMC
                    h = 224
                    w = 224
                    validity = self.net(torch.cat((torch.reshape(torch.transpose(flow, 1, 2), (-1, 2, h, w)),
                                                                 torch.reshape(torch.transpose(input_var[:, 5:7, :, :, :], 1, 2), (-1, 2, h, w))), 0),
                                        node = 'D')
                    loss_adv = self.criterion3(validity, torch.cat((fake_var, valid_var), 0))
        else:
            with torch.no_grad():
                input_var = torch.autograd.Variable(data, volatile=True)
                target_var = torch.autograd.Variable(target, volatile=True)
                if self.criterion2 == None:
                    output = self.net(input_var)
                else:
                    if node == 'logit':
                        output = self.net(input_var[:, :5, :, :, :])
                    else:
                        output, flow = self.net(input_var[:, :5, :, :, :], node = 'flow+logit')
        if hasattr(self, 'criterion') and self.criterion is not None \
            and target is not None:
                # self.criterion2 is the cross entropy for advasarial loss
                if self.criterion2 == None or node == 'logit':
                    loss = self.criterion(output, target_var)
                else:
                    loss = self.criterion(output, target_var) 
                    mse = self.criterion2(flow, input_var[:, 5:7, :, :, :])
        else:
            loss = None
        if not (self.criterion2 == None or node == 'logit'):
            if not stage == None:
                return [output], [loss, mse, loss_adv]
            else:
                return [output], [loss, mse]
        else:
            return [output], [loss]


"""
Dynamic model that is able to update itself
"""
class model(static_model):

    def __init__(self,
                 net,
                 criterion,
                 model_prefix='',
                 step_callback=None,
                 step_callback_freq=50,
                 epoch_callback=None,
                 save_checkpoint_freq=1,
                 opt_batch_size=None,
                 criterion2 = None,
                 criterion3 = None,
                 adv = 0,
                 **kwargs):

        # load parameters
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        super(model, self).__init__(net, criterion=criterion,
                                         model_prefix=model_prefix, criterion2=criterion2, criterion3 = criterion3)

        # load optional arguments
        # - callbacks
        self.callback_kwargs = {'epoch': None,
                                'batch': None,
                                'sample_elapse': None,
                                'update_elapse': None,
                                'epoch_elapse': None,
                                'namevals': None,
                                'optimizer_dict': None,
                                'optimizer_mse_dict': None,}

        if not step_callback:
            step_callback = callback.CallbackList(callback.SpeedMonitor(),
                                                  callback.MetricPrinter())
        if not epoch_callback:
            epoch_callback = (lambda **kwargs: None)

        self.step_callback = step_callback
        self.step_callback_freq = step_callback_freq
        self.epoch_callback = epoch_callback
        self.save_checkpoint_freq = save_checkpoint_freq
        self.batch_size=opt_batch_size
        self.adv = adv


    """
    In order to customize the callback function,
    you will have to overwrite the functions below
    """
    def step_end_callback(self):
        # logging.debug("Step {} finished!".format(self.i_step))
        self.step_callback(**(self.callback_kwargs))

    def epoch_end_callback(self):
        self.epoch_callback(**(self.callback_kwargs))
        if self.callback_kwargs['epoch_elapse'] is not None:
            logging.info("Epoch [{:d}]   time cost: {:.2f} sec ({:.2f} h)".format(
                    self.callback_kwargs['epoch'],
                    self.callback_kwargs['epoch_elapse'],
                    self.callback_kwargs['epoch_elapse']/3600.))
        if self.callback_kwargs['epoch'] == 0 \
           or ((self.callback_kwargs['epoch']+1) % self.save_checkpoint_freq) == 0:
                
            self.save_checkpoint(epoch=self.callback_kwargs['epoch']+1,
                                 optimizer_state=self.callback_kwargs['optimizer_dict'], 
                                 optimizer_mse_state=self.callback_kwargs['optimizer_mse_dict'])

    """
    Learning rate
    """
    def adjust_learning_rate(self, lr, optimizer, epoch = 0, epoch_thre=0):
        for param_group in optimizer.param_groups:
            if 'lr_mult' in param_group:
                lr_mult = param_group['lr_mult']
            else:
                lr_mult = 1.0
            # set a smaller learning rate for the  main convolutional part of I3D
            # if with args.finetune on (0.2) or set a epoch_threshold (0.5)
            if lr_mult == 0.2 or lr_mult == 0.5:
                if epoch_thre > 0 and epoch + 1 <= epoch_thre:
                    lr_mult = 0.  
                else:
                    if lr_mult == 0.5:
                        lr_mult = 1.0
            param_group['lr'] = lr * lr_mult

    """
    Optimization
    """
    def fit(self, train_iter, optimizer, lr_scheduler,
            eval_iter=None,
            metrics=metric.Accuracy(topk=1),
            epoch_start=0,
            epoch_end=10000,
            iter_size = 1,
            clip_gradient = None,
            optimizer_mse = None,
            optimizer_2 = None,
            optimizer_mse_2 = None,
            optimizer_3 = None,
            lr_scheduler2 = None,
            lr_scheduler3 = None,
            metrics_D = None,
            epoch_thre = 1, score_dir = None, detach = False,
            **kwargs):

        """
        checking
        """
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        assert torch.cuda.is_available(), "only support GPU version"

        """
        start the main loop
        """
        pause_sec = 0.
        optimizer.zero_grad()
        if not optimizer_mse == None:
            optimizer_mse.zero_grad()
            optimizer_2.zero_grad()
            optimizer_mse_2.zero_grad()
            indic = 2
            if not optimizer_3 == None:
                optimizer_3.zero_grad()
        else:
            indic = 1
        i = 0.
        top1 = 0
        note = True
        for i_epoch in range(epoch_start, epoch_end):
            self.callback_kwargs['epoch'] = i_epoch
            epoch_start_time = time.time()

            ###########
            # 1] TRAINING
            ###########
            metrics.reset()
            self.net.train()
            sum_sample_inst = 0
            sum_sample_elapse = 0.
            sum_update_elapse = 0
            batch_start_time = time.time()
            logging.info("Start epoch {:d}:".format(i_epoch))
            for i_batch, (data, target) in enumerate(train_iter):
                self.callback_kwargs['batch'] = i_batch

                update_start_time = time.time()
                # There are two stage for joint training which is controlled by epoch_thre                
                if not optimizer_mse == None and i_epoch == epoch_thre and note:
                    logging.info("Replace first stage optimizer with new random initialized optimizer {} {}".format(optimizer is optimizer_2, optimizer_mse is optimizer_mse_2))
                    optimizer = optimizer_2
                    optimizer_mse = optimizer_mse_2
                    note = False
                
                # for discriminator and generator, we updata their parameters separately when it is even and odd iterations                    
                if not optimizer_3 == None and i_batch % (2 * iter_size) < iter_size:
                    outputs, loss_D = self.forward(data, target, node = 'flow+logit', stage = 'D')
                    # [backward]
                    if len(loss_D) == 1:
                        loss_D[0].backward()
                    else:
                        if i_epoch < 1:
                            loss_t = loss_D[0] + self.adv * loss_D[2]
                        else:
                            loss_t = loss_D[0] + self.adv * loss_D[2]

                        loss_t.backward()

                    if not optimizer_mse == None:
                        if i_epoch + 1 <= epoch_thre:
                            lr = lr_scheduler.update()
                            lr2 = lr_scheduler2.update()
                            lr_d = lr_scheduler3.update()
                            if not detach:
                                lr1 = lr
                            else:
                                lr1 = 0.
                        else:
                            lr = lr_scheduler2.update()
                            lr1 = lr
                        self.adjust_learning_rate(optimizer=optimizer,
                                              lr = lr1, epoch = i_epoch, epoch_thre = epoch_thre)
                        self.adjust_learning_rate(optimizer=optimizer_3,
                                              lr = lr_d)
                    else:
                        self.adjust_learning_rate(optimizer=optimizer,
                                              lr=lr_scheduler.update())

                    i = i + 1

                    if i % iter_size == 0:
                        # to train the network on limited gpus, we accumulate gradients
                        # scale down gradients when iter size is functioning
                        if iter_size != 1:
                            for g in optimizer.param_groups:
                                for p in g['params']:
                                    p.grad /= iter_size
                            if not optimizer_3 == None:
                                for g in optimizer_3.param_groups:
                                    for p in g['params']:
                                        p.grad /= iter_size   

                        optimizer.step()
                        optimizer.zero_grad()
                        optimizer_3.step()
                        optimizer_3.zero_grad()
                        i = 0
                    metrics_D.update([output.data.cpu() for output in outputs],
                               target.cpu(),
                                   [loss_D[0].data.cpu(), loss_D[2].data.cpu()])
                        
                # [forward] making next step
                if optimizer_3 == None or (not optimizer_3 == None and i_batch % (2 * iter_size) >= iter_size):
                    if not optimizer_mse == None:
                        if i_epoch + 1 <= epoch_thre:
                            outputs, losses = self.forward(data, target, node = 'flow+logit', stage = optimizer_3)      
                        else:
                            outputs, losses = self.forward(data, target, node = 'flow+logit', stage = optimizer_3) 
                    else:
                        outputs, losses = self.forward(data, target)

                    # [backward]
                    if len(losses) == 1:
                        losses[0].backward()
                    else:
                        if optimizer_3 == None:
                            if i_epoch < 1:
                                loss_t = losses[0]+losses[1]
                            else:
                                loss_t = losses[0]+losses[1]
                        else:
                            if i_epoch < 1:
                                #loss_t = 0. * losses[0] + losses[1] + self.adv * losses[2]
                                loss_t =  0.*losses[0] + losses[1] + self.adv * losses[2]
                            else:
                                loss_t = losses[0] + losses[1] + self.adv * losses[2]

                        loss_t.backward()
                    

                    if not optimizer_mse == None:
                        if i_epoch + 1 <= epoch_thre:
                            if optimizer_3 == None:
                                lr = lr_scheduler.update()
                            lr2 = lr_scheduler2.update()
                            if not detach:
                                lr1 = lr
                            else:
                                lr1 = 0.
                        else:
                            lr = lr_scheduler2.update()
                            lr1 = lr
                        if optimizer_3 == None:
                            self.adjust_learning_rate(optimizer=optimizer,
                                              lr = lr1, epoch = i_epoch, epoch_thre = epoch_thre)
                        self.adjust_learning_rate(optimizer=optimizer_mse,
                                              lr = lr)
                    else:
                        self.adjust_learning_rate(optimizer=optimizer,
                                              lr=lr_scheduler.update())

                    i = i + 1

                    if i % iter_size == 0:
                        # scale down gradients when iter size is functioning
                        if iter_size != 1:
                            if optimizer_3 == None:
                                for g in optimizer.param_groups:
                                    for p in g['params']:
                                        p.grad /= iter_size
                            if not optimizer_mse == None:
                                for g in optimizer_mse.param_groups:
                                    for p in g['params']:
                                        p.grad /= iter_size

                        if clip_gradient is not None:
                            total_norm = clip_grad_norm(self.parameters(), clip_gradient)
                            if total_norm > clip_gradient:
                                logging.info("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
                            else:
                                total_norm = 0
                        if optimizer_3 == None:
                            optimizer.step()
                            optimizer.zero_grad()
                        if not optimizer_mse == None:
                            optimizer_mse.step()
                            optimizer_mse.zero_grad()
                        i = 0
                    
                        # [evaluation] update train metric
                    metrics.update([output.data.cpu() for output in outputs],
                                   target.cpu(),
                                   [loss.data.cpu() for loss in losses])


                # timing each batch
                sum_sample_elapse += time.time() - batch_start_time
                sum_update_elapse += time.time() - update_start_time
                batch_start_time = time.time()
                sum_sample_inst += data.shape[0]

                if (i_batch % self.step_callback_freq) == 0:
                    # retrive eval results and reset metic
                    self.callback_kwargs['namevals'] = metrics.get_name_value()
                    metrics.reset()
                    # speed monitor
                    self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
                    self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
                    sum_update_elapse = 0
                    sum_sample_elapse = 0
                    sum_sample_inst = 0
                    # callbacks
                    self.step_end_callback()
                    if not optimizer_3 == None:
                        logging.info(metrics_D.get_name_value())
                        callback.MetricPrinter(metrics_D.get_name_value())
                        metrics_D.reset()

            ###########
            # 2] END OF EPOCH
            ###########
            self.callback_kwargs['epoch_elapse'] = time.time() - epoch_start_time
            self.callback_kwargs['optimizer_dict'] = optimizer.state_dict()
            if not optimizer_mse == None:
                self.callback_kwargs['optimizer_mse_dict'] = optimizer_mse.state_dict()
            else:
                self.callback_kwargs['optimizer_mse_dict'] = None
            self.epoch_end_callback()

            ###########
            # 3] Evaluation
            ###########
            if (eval_iter is not None) \
                and ((i_epoch+1) % max(1, int(self.save_checkpoint_freq/2))) == 0:
                logging.info("Start evaluating epoch {:d}:".format(i_epoch))

                metrics.reset()
                self.net.eval()
                sum_sample_elapse = 0.
                sum_sample_inst = 0
                sum_forward_elapse = 0.
                batch_start_time = time.time()
                if not score_dir == None:
                    scores = []
                    label = []
                for i_batch, (data, target) in enumerate(eval_iter):
                    self.callback_kwargs['batch'] = i_batch

                    forward_start_time = time.time()

                    outputs, losses = self.forward(data, target, node = 'flow+logit')
                    if not score_dir == None:
                        softmax = torch.nn.Softmax(dim=1)
                        output = softmax(outputs[0]).data.cpu()
                        scores.append(output)
                        target = target.cpu()
                        label.append(target.reshape([-1, 1]))
                    metrics.update([output.data.cpu() for output in outputs],
                                    target.cpu(),
                                   [loss.data.cpu() for loss in losses])
                    del outputs, losses
                    sum_forward_elapse += time.time() - forward_start_time
                    sum_sample_elapse += time.time() - batch_start_time
                    batch_start_time = time.time()
                    sum_sample_inst += data.shape[0]
                
                # evaluation callbacks
                self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
                self.callback_kwargs['update_elapse'] = sum_forward_elapse / sum_sample_inst
                self.callback_kwargs['namevals'] = metrics.get_name_value()
                
                if not score_dir == None and top1 < metrics.get_name_value()[indic][0][1]:
                    print(np.concatenate(scores, axis = 0).shape, np.concatenate(label, axis = 0).shape)
                    
                    np.savez(score_dir,scores=np.concatenate(scores, axis = 0), labels=np.concatenate(label, axis = 0), \
                             top1 = metrics.get_name_value()[indic][0][1])
                
                    top1 = metrics.get_name_value()[indic][0][1]
                    logging.info("save new best score with top1 {}".format(top1))
                self.step_end_callback()
        logging.info("Optimization done!")
