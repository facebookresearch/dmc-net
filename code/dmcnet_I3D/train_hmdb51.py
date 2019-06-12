"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import json
import socket
import logging
import argparse

import torch
import torch.nn.parallel
import torch.distributed as dist

import dataset
from train_model import train_model
from network.symbol_builder import get_symbol


parser = argparse.ArgumentParser(description="DMC-Net Parser")
# debug
parser.add_argument('--debug-mode', type=bool, default=True,
                    help="print all setting for debugging.")
# io
parser.add_argument('--dataset', default='UCF101', choices=['UCF101', 'HMDB51', 'Kinetics'],
                    help="path to dataset")
parser.add_argument('--split', type = int, default=1, 
                    help="which split to train on")
parser.add_argument('--clip-length',type=int, default=16,
                    help="define the length of each input sample.")
parser.add_argument('--train-frame-interval', type=int, default=2,
                    help="define the sampling interval between frames.")
parser.add_argument('--val-frame-interval', type=int, default=2,
                    help="define the sampling interval between frames.")
parser.add_argument('--task-name', type=str, default='',
                    help="name of current task, leave it empty for using folder name")
parser.add_argument('--model-dir', type=str, default="./exps/models",
                    help="set logging file.")
parser.add_argument('--log-file', type=str, default="",
                    help="set logging file.")
parser.add_argument('--accumulate', type=int, default=1,
                    help="accumulate mv and res")
parser.add_argument('--mv-minmaxnorm', type=int, default=0,
                    help="minmaxnorm for mv")
parser.add_argument('--mv-loadimg', type=int, default=0,
                    help="load img mv")
parser.add_argument('--detach', type=int, default=0,
                    help="whether not update i3d")
parser.add_argument('--ds_factor', type=int, default=16,
                    help="downsampling the flow by ds_factor")
# device
parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7",
                    help="define gpu id")
# algorithm
parser.add_argument('--network', type=str, default='MFNet_3D',
                    choices=['MFNet_3D', 'I3D'],
                    help="chose the base network")
parser.add_argument('--arch-estimator', type=str, default = None,
                    choices=['DenseNet','DenseNetSmall', 'DenseNetTiny'],
                    help="chose the generator")
parser.add_argument('--arch-d', type=str, default=None,
                    help="chose the D")
# initialization with priority (the next step will overwrite the previous step)
# - step 1: random initialize
# - step 2: load the 2D pretrained model if `pretrained_2d' is True
# - step 3: load the 3D pretrained model if `pretrained_3d' is defined
# - step 4: resume if `resume_epoch' >= 0
parser.add_argument('--pretrained_2d', type=bool, default=False,
                    help="load default 2D pretrained model.")
parser.add_argument('--pretrained_3d', type=str, 
                    default='./network/pretrained/MFNet3D_Kinetics-400_72.8.pth',
                    help="load default 3D pretrained model.")
parser.add_argument('--new_classifier', type=bool, default=False,
                    help="whether use mode_flow to initialize classifier weights")
parser.add_argument('--resume-epoch', type=int, default=-1,
                    help="resume train")
# flow+mp4 is the modality we used for generating DMC
parser.add_argument('--modality', type=str, default='rgb',
                    choices=['rgb', 'flow', 'mv', 'res', 'flow+mp4', 'I'],
                    help="chose input type")
parser.add_argument('--drop-out', type=float, default=0.5,
                    help="drop-out probability")
parser.add_argument('--adv', type=float, default=0.,
                    help="weight for adversirial loss")
# optimization
parser.add_argument('--epoch-thre', type=int, default=1,
                    help="the epoch classifier begins to be optimized when with gen")
parser.add_argument('--optimizer', type=str, default='sgd',
                    choices=['sgd', 'adam'],
                    help="optimizer")
parser.add_argument('--fine_tune', type=int, default=1,
                    help="apply different learning rate for different layers")
parser.add_argument('--batch-size', type=int, default=32,
                    help="batch size")
parser.add_argument('--iter-size', type=int, default=1,
                    help="iteration size which is for accumalation of gradients")
parser.add_argument('--lr-base', type=float, default=0.005,
                    help="learning rate")
parser.add_argument('--lr-base2', type=float, default=0.002,
                    help="learning rate for stage 2")
parser.add_argument('--lr-d', type=float, default=None,
                    help="learning rate for discriminator")
parser.add_argument('--lr-steps', type=list, default=[int(1e4*x) for x in [3.5, 6, 8.5, 11, 13.5, 16]],
                    help="number of samples to pass before changing learning rate") # 1e6 million
#parser.add_argument('--lr-steps', type=list, default=[int(1e4*x) for x in [4.5, 7, 9.5, 12, 14.5, 17]],
#                    help="number of samples to pass before changing learning rate") # 1e6 million
#parser.add_argument('--lr-steps', type=list, default=[int(1e4*x) for x in [10, 20, 30, 40, 50, 60]],
#                    help="number of samples to pass before changing learning rate") # 1e6 million
parser.add_argument('--lr-factor', type=float, default=0.1,
                    help="reduce the learning with factor")
parser.add_argument('--save-frequency', type=float, default=1,
                    help="save once after N epochs")
parser.add_argument('--end-epoch', type=int, default=50,
                    help="maxmium number of training epoch")
parser.add_argument('--random-seed', type=int, default=1,
                    help='random seed (default: 1)')

def autofill(args):
    # customized
    if not args.task_name:
        args.task_name = os.path.basename(os.getcwd())
    if not args.log_file:
        if os.path.exists("./exps/logs"):
            args.log_file = "./exps/logs/{}_at-{}.log".format(args.task_name, socket.gethostname())
        else:
            args.log_file = ".{}_at-{}.log".format(args.task_name, socket.gethostname())
    # fixed
    args.model_prefix = os.path.join(args.model_dir, args.task_name)
    args.score_dir = './exps/score' + '/{}_{}/'.format(args.dataset, args.split) + args.task_name
    return args

def set_logger(log_file='', debug_mode=False):
    if log_file:
        if not os.path.exists("./"+os.path.dirname(log_file)):
            os.makedirs("./"+os.path.dirname(log_file))
        handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    else:
        handlers = [logging.StreamHandler()]

    """ add '%(filename)s:%(lineno)d %(levelname)s:' to format show source file """
    logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO,
                format='%(asctime)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers = handlers)

if __name__ == "__main__":

    # set args
    args = parser.parse_args()
    args = autofill(args)

    set_logger(log_file=args.log_file, debug_mode=args.debug_mode)
    logging.info("Using pytorch {} ({})".format(torch.__version__, torch.__path__))
    logging.info("Start training with args:\n" +
                 json.dumps(vars(args), indent=4, sort_keys=True))

    # set device states
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # before using torch
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)


    # load dataset related configuration
    dataset_cfg = dataset.get_config(name=args.dataset)

    # creat model with all parameters initialized
    net, input_conf = get_symbol(name=args.network,
                     pretrained=args.pretrained_2d if args.resume_epoch < 0 else None,
                     modality = args.modality,
                     drop_out = args.drop_out,
                     arch_estimator = args.arch_estimator,
                     arch_d = args.arch_d,
                     print_net = False,
                     **dataset_cfg)

    # training
    kwargs = {}
    kwargs.update(dataset_cfg)
    kwargs.update({'input_conf': input_conf})
    kwargs.update(vars(args))
    train_model(args.network, sym_net=net, optim = args.optimizer, **kwargs)
