"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import logging

import torch

from . import video_sampler as sampler
from . import video_transforms as transforms
from .video_iterator import VideoIter

def get_hmdb51(data_root='./dataset/HMDB51',
               clip_length=8,
               train_interval=2,
               val_interval=2,
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225],
               seed=torch.distributed.get_rank() if torch.distributed._initialized else 0,
               modality = 'rgb',
               split = 1,
               net_name = 'MFNet_3D',
               accumulate = True,
               ds_factor = 16,
               mv_minmaxnorm = False,
               mv_loadimg = False,
               **kwargs):
    """ data iter for ucf-101
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    normalize = transforms.Normalize(mean=mean, std=std)
    if modality in ['rgb', 'I']:
        tran = transforms.Compose([
                                         transforms.RandomScale(make_square=True,
                                                                aspect_ratio=[0.8, 1./0.8],
                                                                slen=[224, 288]),
                                         transforms.RandomCrop((224, 224)), # insert a resize if needed
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomHLS(vars=[15, 35, 25]),
                                         transforms.ToTensor(modality),
                                         normalize,
                                      ],
                                      aug_seed=(seed+1))
    elif modality in ['flow', 'mv', 'res', 'flow+mp4']:
        tran = transforms.Compose([
                                         transforms.RandomCrop((224, 224)), # insert a resize if needed
                                         transforms.RandomHorizontalFlip(modality),
                                         transforms.ToTensor(modality, ds_factor),
                                         normalize,
                                      ],
                                      aug_seed=(seed+1))
    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))
    train = VideoIter(video_prefix='/HMDB51/fb/TSN_input/', #change accordingly
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'hmdb51_split{}_train.txt'.format(split)),
                      sampler=train_sampler,
                      force_color=True,
                      video_transform=tran,
                      name='train',
                      cached_info_path = os.path.join(data_root, 'raw',  'list_cvt', 'hmdb51_split{}_train_info.txt'.format(split)),
                      shuffle_list_seed=(seed+2),
                      check_video = True,
                      load_from_img = True,
                      modality = modality, accumulate = accumulate, ds_factor = ds_factor, mv_minmaxnorm = mv_minmaxnorm,mv_loadimg=mv_loadimg)
    if net_name == 'I3D':
        tran = transforms.Compose([
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(modality),
                                         normalize,
                                      ])
    elif net_name == 'MFNet_3D':
        tran = transforms.Compose([
                                         transforms.Resize((256, 256)),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(modality),
                                         normalize,
                                      ])
    val_sampler   = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)
    val   = VideoIter(video_prefix='/HMDB51/fb/TSN_input/', #change accordingly
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'hmdb51_split{}_test.txt'.format(split)),
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=tran,
                      name='test',
                      cached_info_path = os.path.join(data_root, 'raw',  'list_cvt', 'hmdb51_split{}_test_info.txt'.format(split)),
                      check_video = True,
                      load_from_img = True,
                      modality = modality, accumulate = accumulate, ds_factor = ds_factor, mv_minmaxnorm = mv_minmaxnorm,mv_loadimg=mv_loadimg)

    return (train, val)

def get_ucf101(data_root='./dataset/UCF101',
               clip_length=8,
               train_interval=2,
               val_interval=2,
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225],
               seed=torch.distributed.get_rank() if torch.distributed._initialized else 0,
               modality = 'rgb',
               split = 1,
               net_name = 'MFNet_3D',
               accumulate = True,
               ds_factor = 16,
               mv_minmaxnorm = False,
               mv_loadimg=False,
               **kwargs):
    """ data iter for ucf-101
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    normalize = transforms.Normalize(mean=mean, std=std)

    if modality  in ['rgb', 'I']:
        tran = transforms.Compose([
                                         transforms.RandomScale(make_square=True,
                                                                aspect_ratio=[0.8, 1./0.8],
                                                                slen=[224, 288]),
                                         transforms.RandomCrop((224, 224)), # insert a resize if needed
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomHLS(vars=[15, 35, 25]),
                                         transforms.ToTensor(modality),
                                         normalize,
                                      ],
                                      aug_seed=(seed+1))
    elif modality in ['flow', 'mv', 'res', 'flow+mp4']:
        tran = transforms.Compose([
                                         transforms.RandomCrop((224, 224)), # insert a resize if needed
                                         transforms.RandomHorizontalFlip(modality),
                                         transforms.ToTensor(modality, ds_factor),
                                         normalize,
                                      ],
                                      aug_seed=(seed+1))
    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))
    train = VideoIter(video_prefix='/HMDB51/fb/TSN_input/', #change accordingly
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'trainlist0{}.txt'.format(split)),
                      sampler=train_sampler,
                      force_color=True,
                      video_transform=tran,
                      name='train',
                      shuffle_list_seed=(seed+2),
                      cached_info_path = os.path.join(data_root, 'raw',  'list_cvt', 'ucf101_split{}_train_info.txt'.format(split)),
                      check_video = True,
                      load_from_img = True,
                      modality = modality, accumulate = accumulate, ds_factor = ds_factor, mv_minmaxnorm = mv_minmaxnorm,mv_loadimg=mv_loadimg)

    if net_name == 'I3D':
        tran = transforms.Compose([
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(modality),
                                         normalize,
                                      ])
    elif net_name == 'MFNet_3D':
        tran = transforms.Compose([
                                         transforms.Resize((256, 256)),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(modality),
                                         normalize,
                                      ])
    val_sampler   = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)
    val   = VideoIter(video_prefix='/HMDB51/fb/TSN_input/', #change accordingly
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'testlist0{}.txt'.format(split)),
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=tran,
                      name='test',
                      cached_info_path = os.path.join(data_root, 'raw',  'list_cvt', 'ucf101_split{}_test_info.txt'.format(split)),
                      check_video = True,
                      load_from_img = True,
                      modality = modality, accumulate = accumulate, ds_factor = ds_factor, mv_minmaxnorm = mv_minmaxnorm,mv_loadimg=mv_loadimg
                      )

    return (train, val)


def get_kinetics(data_root='./dataset/Kinetics',
                 clip_length=8,
                 train_interval=2,
                 val_interval=2,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 seed=torch.distributed.get_rank() if torch.distributed._initialized else 0,
                 **kwargs):
    """ data iter for kinetics
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    normalize = transforms.Normalize(mean=mean, std=std)

    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))
    train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data', 'train_avi-x256'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_train_w-missed-v1_avi.txt'),
                      sampler=train_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.RandomScale(make_square=True,
                                                                aspect_ratio=[0.8, 1./0.8],
                                                                slen=[224, 288]),
                                         transforms.RandomCrop((224, 224)), # insert a resize if needed
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomHLS(vars=[15, 35, 25]),
                                         transforms.ToTensor(),
                                         normalize,
                                      ],
                                      aug_seed=(seed+1)),
                      name='train',
                      shuffle_list_seed=(seed+2),
                      )

    val_sampler   = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)
    val   = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data', 'val_avi-x256'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_val_w-missed-v1_avi.txt'),
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.Resize((256, 256)),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         normalize,
                                      ]),
                      name='test',
                      )
    return (train, val)



def creat(name, batch_size, num_workers=8, **kwargs):

    if name.upper() == 'UCF101':
        train, val = get_ucf101(**kwargs)
    elif name.upper() == 'HMDB51':
        train, val = get_hmdb51(**kwargs)
    elif name.upper() == 'KINETICS':
        train, val = get_kinetics(**kwargs)
    else:
        assert NotImplementedError("iter {} not found".format(name))


    train_loader = torch.utils.data.DataLoader(train,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(val,
        batch_size = batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False)

    return (train_loader, val_loader)
