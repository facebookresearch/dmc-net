"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""Run testing given a trained model."""

import argparse
import time
import os

from dataset import CoviarDataSet
from model import Model
from transforms import GroupCenterCrop
from transforms import GroupOverSample
from transforms import GroupScale

import numpy as np
import torch.nn.parallel
import torch.optim
import torchvision


parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--data-name', type=str, choices=['ucf101', 'hmdb51', 'kinetics400'])
parser.add_argument('--representation', type=str, choices=['iframe', 'residual', 'mv', 'flow'])
parser.add_argument('--no-accumulation', action='store_true',
                    help='disable accumulation of motion vectors and residuals.')
parser.add_argument('--new_length', type=int, default=1,
                    help='number of MV/OF stacked to be processed together.')
parser.add_argument('--use_databn', type=int, default=1,
                    help='add databn for mv, residual, flow or not.')
parser.add_argument('--flow_ds_factor', type=int, default=0,
                    help='flow downsample factor.')
parser.add_argument('--upsample_interp', type=bool, default=False,
                    help='upsample via interpolation or not.')
parser.add_argument('--data-root', type=str)
parser.add_argument('--flow-root', type=str, help='directory of storing pre-extracted optical flow images')
parser.add_argument('--data-flow', type=str, default='tvl1')
parser.add_argument('--test-list', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--batch-size', default=1, type=int, help='batch size.')
parser.add_argument('--arch', type=str)
parser.add_argument('--arch_estimator', type=str, default="ContextNetwork", help='estimator architecture.')
parser.add_argument('--arch_d', type=str, default="Discriminator", help='discriminator architecture.')
parser.add_argument('--save-scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--test-crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of workers for data loader.')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--gop', type=int, default=12, help='size of GOP.')
parser.add_argument('--viz', type=bool, default=False, help='visualize or not.')
parser.add_argument('--gen_flow_or_delta', type=int, default=0, help='0: generate flow; 1: generate flow delta')
parser.add_argument('--gen_flow_ds_factor', type=int, default=0, help='the downsample factor used in generating flow of small size')
parser.add_argument('--att', type=int, default=0, help='0: no attention; 1: pixel-level attention.')
parser.add_argument('--mv_minmaxnorm', type=int, default=1,
                    help='use min max normalization for mv value to map from 128+-20 to 128+-127 something.')

args = parser.parse_args()

if args.data_name == 'ucf101':
    num_class = 101
elif args.data_name == 'hmdb51':
    num_class = 51
elif args.data_name == 'kinetics400':
    num_class = 400
else:
    raise ValueError('Unknown dataset '+args.data_name)


def main():
    # define the whole model network architecture
    net = Model(num_class, args.test_segments, args.representation,
                base_model=args.arch,
                new_length=args.new_length,
                use_databn=args.use_databn,
                gen_flow_or_delta=args.gen_flow_or_delta,
                gen_flow_ds_factor=args.gen_flow_ds_factor,
                arch_estimator=args.arch_estimator,
                arch_d=args.arch_d,
                att=args.att)

    # load the trained model
    checkpoint = torch.load(args.weights, map_location=lambda storage, loc: storage)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict, strict=False)

    # setup the data loader
    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(net.crop_size),
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(net.crop_size, net.scale_size)
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported, but got {}.".format(args.test_crops))

    data_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.flow_root,
            args.data_name,
            video_list=args.test_list,
            num_segments=args.test_segments,
            representation=args.representation,
            new_length=args.new_length,
            flow_ds_factor=args.flow_ds_factor,
            upsample_interp=args.upsample_interp,
            transform=cropping,
            is_train=False,
            accumulate=(not args.no_accumulation),
            gop=args.gop,
            flow_folder=args.data_flow,
            mv_minmaxnorm=args.mv_minmaxnorm,
            viz=args.viz
            ),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))

    net.cuda(devices[0])
    #net.base_model.cuda(devices[-1])
    net = torch.nn.DataParallel(net, device_ids=devices)

    # switch to inference model and start to iterate over the test set
    net.eval()

    total_num = len(data_loader.dataset)
    output = []

    # process each video to obtain its predictions
    def forward_video(input_mv, input_residual, att=0):
        input_mv_var = torch.autograd.Variable(input_mv, volatile=True)
        input_residual_var = torch.autograd.Variable(input_residual, volatile=True)
        if att == 0:
            scores, validity, gen_flow = net(input_mv_var, input_residual_var)
        if att == 1:
            scores, validity, gen_flow, att_flow = net(input_mv_var, input_residual_var)
        scores = scores.view((-1, args.test_segments * args.test_crops) + scores.size()[1:])
        scores = torch.mean(scores, dim=1)
        if att == 0:
            return scores.data.cpu().numpy().copy(), validity.data.cpu().numpy().copy(), gen_flow
        if att == 1:
            return scores.data.cpu().numpy().copy(), validity.data.cpu().numpy().copy(), gen_flow, att_flow

    proc_start_time = time.time()

    # iterate over the whole test set
    for i, (input_flow, input_mv, input_residual, label) in enumerate(data_loader):
        input_mv = input_mv.cuda(args.gpus[-1], async=True)
        input_residual = input_residual.cuda(args.gpus[0], async=True)
        input_flow = input_flow.cuda(args.gpus[-1], async=True)

        # print("input_flow shape:")
        # print(input_flow.shape) # torch.Size([batch_size, num_crops*num_segments, 2, 224, 224])
        # print("input_flow type:")  # print(input_flow.type())  # torch.cuda.FloatTensor
        if args.att == 0:
            video_scores, validity, gen_flow = forward_video(input_mv, input_residual)
        if args.att == 1:
            video_scores, validity, gen_flow, att_flow = forward_video(input_mv, input_residual, args.att)
        output.append((video_scores, label[0], validity))
        cnt_time = time.time() - proc_start_time
        if (i + 1) % 100 == 0:
            print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                            total_num,
                                                                            float(cnt_time) / (i+1)))

    video_pred = [np.argmax(x[0]) for x in output]
    video_labels = [x[1] for x in output]
    video_validity = [np.argmax(x[2]) for x in output]

    print('Accuracy cls {:.02f}% ({})'.format(
        float(np.sum(np.array(video_pred) == np.array(video_labels))) / len(video_pred) * 100.0,
        len(video_pred)))

    print('Accuracy adv G {:.02f}% ({})'.format(
        float(np.sum(np.array(video_validity))) / len(video_validity) * 100.0,
        len(video_validity)))

    if args.save_scores is not None:

        name_list = [x.strip().split()[0] for x in open(args.test_list)]
        order_dict = {e:i for i, e in enumerate(sorted(name_list))}

        reorder_output = [None] * len(output)
        reorder_label = [None] * len(output)
        reorder_name = [None] * len(output)

        for i in range(len(output)):
            idx = order_dict[name_list[i]]
            reorder_output[idx] = output[i]
            reorder_label[idx] = video_labels[i]
            reorder_name[idx] = name_list[i]

        np.savez(args.save_scores, scores=reorder_output, labels=reorder_label, names=reorder_name)


if __name__ == '__main__':
    main()
