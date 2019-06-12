"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""Run training."""

import shutil
import time
import numpy as np
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision

from dataset import CoviarDataSet
from model import Model
from train_options import parser
from transforms import GroupCenterCrop
from transforms import GroupScale

SAVE_FREQ = 40
PRINT_FREQ = 20
best_prec1 = 0


def main():
    # loading input arguments for training
    global args
    global best_prec1
    global start_epoch
    start_epoch = 0
    args = parser.parse_args()

    print('Training arguments:')
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))

    if args.data_name == 'ucf101':
        num_class = 101
    elif args.data_name == 'hmdb51':
        num_class = 51
    elif args.data_name == 'kinetics400':
        num_class = 400
    else:
        raise ValueError('Unknown dataset ' + args.data_name)

    # define the model architecture
    model = Model(num_class, args.num_segments, args.representation,
                  base_model=args.arch,
                  new_length=args.new_length,
                  use_databn=args.use_databn,
                  gen_flow_or_delta=args.gen_flow_or_delta,
                  gen_flow_ds_factor=args.gen_flow_ds_factor,
                  arch_estimator=args.arch_estimator,
                  att=args.att)
    print(model)

    # load the pre-trained model
    if args.weights is not None:
        checkpoint = torch.load(args.weights, map_location=lambda storage, loc: storage)
        print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
        base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
        model.load_state_dict(base_dict, strict=False)

    # define the data loader for reading training data
    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.flow_root,
            args.data_name,
            video_list=args.train_list,
            num_segments=args.num_segments,
            representation=args.representation,
            new_length=args.new_length,
            flow_ds_factor=args.flow_ds_factor,
            upsample_interp=args.upsample_interp,
            transform=model.get_augmentation(),
            is_train=True,
            accumulate=(not args.no_accumulation),
            gop=args.gop,
            flow_folder=args.data_flow,
            mv_minmaxnorm=args.mv_minmaxnorm,
            ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # define the data loader for reading val data
    val_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.flow_root,
            args.data_name,
            video_list=args.test_list,
            num_segments=args.num_segments,
            representation=args.representation,
            new_length=args.new_length,
            flow_ds_factor=args.flow_ds_factor,
            upsample_interp=args.upsample_interp,
            transform=torchvision.transforms.Compose([
                GroupScale(int(model.scale_size)),
                GroupCenterCrop(model.crop_size),
                ]),
            is_train=False,
            accumulate=(not args.no_accumulation),
            gop=args.gop,
            flow_folder=args.data_flow,
            mv_minmaxnorm=args.mv_minmaxnorm,
            ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda(args.gpus[0])
    cudnn.benchmark = True

    # define optimizer and specify the corresponding parameters
    params_dict = dict(model.named_parameters())
    params_cls = []
    params_gf = []
    for key, value in params_dict.items():
        if 'base_model' in key:
            decay_mult = 0.0 if 'bias' in key else 1.0
            lr_mult = args.lr_cls_mult # for cls, just finetune. if '.fc.' in key: lr_mult = 1.0
            params_cls += [{'params': value, 'lr': args.lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult}]
        if 'gen_flow_model' in key:
            decay_mult = 0.0 if 'bias' in key else 1.0
            lr_mult = args.lr_mse_mult # for cls, just finetune. if '.fc.' in key: lr_mult = 1.0
            params_gf += [{'params': value, 'lr': args.lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult}]

    optimizer_cls = torch.optim.Adam(
        params_cls,
        weight_decay=args.weight_decay,
        eps=0.001)

    optimizer_gf = torch.optim.Adam(
        params_gf,
        weight_decay=args.weight_decay,
        eps=0.001)

    # resume training from previous checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer_cls' in checkpoint.keys():
                optimizer_cls.load_state_dict(checkpoint['optimizer_cls'])
                optimizer_gf.load_state_dict(checkpoint['optimizer_gf'])
                def load_opt_update_cuda(optimizer, cuda_id):
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda(cuda_id)
                load_opt_update_cuda(optimizer_cls, args.gpus[0])
                load_opt_update_cuda(optimizer_gf, args.gpus[0])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    # define several loss functions
    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpus[0])
    if args.loss_mse == 'MSELoss':
        criterion_mse = torch.nn.MSELoss().cuda(args.gpus[0])
    elif args.loss_mse == 'SmoothL1Loss':
        criterion_mse = torch.nn.SmoothL1Loss().cuda(args.gpus[0])
    elif args.loss_mse == 'L1':
        criterion_mse = torch.nn.L1Loss().cuda(args.gpus[0])

    # finally done with setup and start to train model
    for epoch in range(start_epoch, args.epochs):
        # determine the learning rate for the current epoch
        cur_lr_cls = adjust_learning_rate(optimizer_cls, epoch, args.lr_steps, args.lr_decay, freeze=True, epoch_thre=args.epoch_thre)
        cur_lr_gf = adjust_learning_rate(optimizer_gf, epoch, args.lr_steps, args.lr_decay)

        # perform training
        print("current epoch freeze?: {}".format(str(epoch < args.epoch_thre)))
        train(train_loader, model, criterion, criterion_mse, optimizer_cls,
            optimizer_gf, epoch, cur_lr_cls, cur_lr_gf, args.lr_cls, args.lr_mse, args.att, freeze=(epoch < args.epoch_thre))

        # perform validation if needed
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, criterion_mse, args.lr_cls, args.lr_mse, args.att)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if is_best or epoch % SAVE_FREQ == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer_cls': optimizer_cls.state_dict(),
                        'optimizer_gf': optimizer_gf.state_dict(),
                    },
                    is_best,
                    filename='checkpoint.pth.tar')


# define the function of training for one epoch
def train(train_loader, model, criterion, criterion_mse, optimizer, optimizer_gf, epoch, cur_lr_cls, cur_lr_gf, lr_cls, lr_mse, att, freeze=False):
    # init meter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_gf = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to training mode
    model.train()

    end = time.time()

    # iter over each batch
    for i, (input_flow, input_mv, input_residual, target) in enumerate(train_loader):

        data_time.update(time.time() - end)

        # prepare input data consisting of mv input and flow input
        target = target.cuda(args.gpus[0], async=True)
        input_mv = input_mv.cuda(args.gpus[0], async=True)
        input_residual = input_residual.cuda(args.gpus[0], async=True)
        input_flow = input_flow.cuda(args.gpus[0], async=True)
        input_flow = input_flow.view((-1, ) + input_mv.size()[-3:])
        input_mv_var = torch.autograd.Variable(input_mv)
        input_residual_var = torch.autograd.Variable(input_residual)
        target_var = torch.autograd.Variable(target)

        # forward
        output, gen_flow = model(input_mv_var, input_residual_var)

        # TSN classification loss
        output = output.view((-1, args.num_segments) + output.size()[1:])
        output = torch.mean(output, dim=1)
        loss_cls = criterion(output, target_var)

        # add flow reconstruction mse loss
        input_flow_var = torch.autograd.Variable(input_flow)
        loss_mse = criterion_mse(gen_flow, input_flow_var)   # input, target

        # total loss
        loss = loss_cls * lr_cls + loss_mse * lr_mse

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input_flow.size(0))
        losses_cls.update(loss_cls.data[0], input_flow.size(0))
        losses_gf.update(loss_mse.data[0], input_flow.size(0))
        top1.update(prec1[0], input_flow.size(0))
        top5.update(prec5[0], input_flow.size(0))

        # backward update
        optimizer.zero_grad()
        optimizer_gf.zero_grad()
        if freeze == True:
            loss_mse = loss_mse * lr_mse
            loss_mse.backward()
        else:
            loss.backward()
            optimizer.step()
        optimizer_gf.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print(('Epoch: [{0}][{1}/{2}], lr_gf: {lr:.7f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                   'loss_mse {loss_mse.val:.4f} ({loss_mse.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader),
                       batch_time=batch_time,
                       data_time=data_time,
                       loss=losses,
                       loss_cls=losses_cls,
                       loss_mse=losses_gf,
                       top1=top1,
                       top5=top5,
                       lr=cur_lr_gf)))


# define the function of performing validation
def validate(val_loader, model, criterion, criterion_mse, lr_cls, lr_mse, att):
    # init meter
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_gf = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to eval mode
    model.eval()

    end = time.time()

    # iter over each batch
    for i, (input_flow, input_mv, input_residual, target) in enumerate(val_loader):

        # prepare input data consisting of mv input and flow input
        target = target.cuda(args.gpus[0], async=True)
        input_mv = input_mv.cuda(args.gpus[0], async=True)
        input_residual = input_residual.cuda(args.gpus[0], async=True)
        input_flow = input_flow.cuda(args.gpus[0], async=True)
        input_flow = input_flow.view((-1, ) + input_mv.size()[-3:])
        input_mv_var = torch.autograd.Variable(input_mv, volatile=True)
        input_residual_var = torch.autograd.Variable(input_residual, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # forward
        if att == 0:
            output, gen_flow = model(input_mv_var, input_residual_var)
        elif att == 1:
            output, gen_flow, att_flow = model(input_mv_var, input_residual_var)

        # TSN classification loss
        output = output.view((-1, args.num_segments) + output.size()[1:])
        output = torch.mean(output, dim=1)
        loss_cls = criterion(output, target_var)

        # add flow reconstruction mse loss
        input_flow_var = torch.autograd.Variable(input_flow, volatile=True)
        if att == 0:
            loss_mse = criterion_mse(gen_flow, input_flow_var)   # input, target
        elif att == 1:
            loss_mse = criterion_mse(att_flow * gen_flow, att_flow * input_flow_var)

        # total loss
        loss = loss_cls * lr_cls + loss_mse * lr_mse

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input_flow.size(0))
        losses_cls.update(loss_cls.data[0], input_flow.size(0))
        losses_gf.update(loss_mse.data[0], input_flow.size(0))
        top1.update(prec1[0], input_flow.size(0))
        top5.update(prec5[0], input_flow.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                   'loss_mse {loss_mse.val:.4f} ({loss_mse.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader),
                       batch_time=batch_time,
                       loss=losses,
                       loss_cls=losses_cls,
                       loss_mse=losses_gf,
                       top1=top1,
                       top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg


def save_checkpoint(state, is_best, filename):
    filename = '_'.join((args.model_prefix, args.representation.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.model_prefix, args.representation.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay, freeze=False, epoch_thre=500):
    decay = lr_decay ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    wd = args.weight_decay
    if epoch < epoch_thre and freeze:
        lr = 0
        wd = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = wd * param_group['decay_mult']
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
