"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""Training options."""

import argparse

parser = argparse.ArgumentParser(description="CoViAR")

# Data.
parser.add_argument('--data-name', type=str, choices=['ucf101', 'hmdb51', 'kinetics400'],
                    help='dataset name.')
parser.add_argument('--data-root', type=str,
                    help='root of data directory.')
parser.add_argument('--flow-root', type=str, 
                    help='directory of storing pre-extracted optical flow images.')
parser.add_argument('--data-flow', type=str, default='tvl1',
                    help='root of data directory.')
parser.add_argument('--train-list', type=str,
                    help='training example list.')
parser.add_argument('--test-list', type=str,
                    help='testing example list.')
parser.add_argument('--gop', type=int, default=12,
                    help='size of GOP.')

# Model.
parser.add_argument('--representation', type=str, choices=['iframe', 'mv', 'residual', 'flow'],
                    help='data representation.')
parser.add_argument('--arch', type=str, default="resnet152",
                    help='base architecture.')
parser.add_argument('--arch_estimator', type=str, default="ContextNetwork",
                    help='estimator architecture.')
parser.add_argument('--num_segments', type=int, default=3,
                    help='number of TSN segments.')
parser.add_argument('--no-accumulation', action='store_true',
                    help='disable accumulation of motion vectors and residuals.')
parser.add_argument('--new_length', type=int, default=1,
                    help='number of MV/OF stacked to be processed together.')
parser.add_argument('--flow_ds_factor', type=int, default=0,
                    help='flow downsample factor.')
parser.add_argument('--gen_flow_ds_factor', type=int, default=0,
                    help='the downsample factor used in generating flow of small size')
parser.add_argument('--upsample_interp', type=bool, default=False,
                    help='upsample via interpolation or not.')
parser.add_argument('--use_databn', type=int, default=1,
                    help='add data batchnorm for mv, residual, flow or not. 1: yes; 0: no.')
parser.add_argument('--gen_flow_or_delta', type=int, default=0,
                    help='0: generate flow; 1: generate flow delta.')
parser.add_argument('--att', type=int, default=0,
                    help='0: no attention; 1: pixel-level attention.')
parser.add_argument('--mv_minmaxnorm', type=int, default=0,
                    help='use min max normalization for mv value to map from 128+-20 to 128+-127 something.')

# Training.
parser.add_argument('--weights', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--epochs', default=500, type=int,
                    help='number of training epochs.')
parser.add_argument('--epoch-thre', default=500, type=int,
                    help='number of training epochs.')
parser.add_argument('--batch-size', default=40, type=int,
                    help='batch size.')
parser.add_argument('--lr', default=0.001, type=float,
                    help='base learning rate.')
parser.add_argument('--lr-cls', default=1, type=float,
                    help='cls loss weight.')
parser.add_argument('--loss-mse', default='MSELoss', type=str)
parser.add_argument('--lr-mse', default=0.1, type=float,
                    help='mse loss weight.')
parser.add_argument('--lr_cls_mult', default=0.01, type=float, help='cls learning multiplier.')
parser.add_argument('--lr_mse_mult', default=0.01, type=float, help='mse learning multiplier.')
parser.add_argument('--lr-steps', default=[200, 300, 400], type=float, nargs="+",
                    help='epochs to decay learning rate.')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    help='lr decay factor.')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay.')

# Log.
parser.add_argument('--eval-freq', default=5, type=int,
                    help='evaluation frequency (epochs).')
parser.add_argument('--workers', default=8, type=int,
                    help='number of data loader workers.')
parser.add_argument('--model-prefix', type=str, default="model",
                    help="prefix of model name.")
parser.add_argument('--gpus', nargs='+', type=int, default=None,
                    help='gpu ids.')
