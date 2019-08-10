"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors,
or residuals) for training or testing.
"""

import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data

from coviar import get_num_frames
from coviar import load
from transforms import color_aug
from PIL import Image
from skimage.measure import block_reduce
from scipy import interpolate

import logging
logging.basicConfig(level=logging.DEBUG)

GOP_SIZE = 12

# '/projects/eventnet/dataset/HMDB51/fb/TSN_input/'

def video_path_to_flow_path(flow_root, video_path):
    # example:
    tmp = video_path.split('/')
    return os.path.join(flow_root, tmp[-2], tmp[-1][:-4])

def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)


def get_seg_range(n, num_segments, seg, representation):
    if representation in ['residual', 'mv', 'flow']:
        n -= 1

    seg_size = float(n - 1) / num_segments
    seg_begin = int(np.round(seg_size * seg))
    seg_end = int(np.round(seg_size * (seg+1)))
    if seg_end == seg_begin:
        seg_end = seg_begin + 1

    if representation in ['residual', 'mv', 'flow']:
        # Exclude the 0-th frame, because it's an I-frmae.
        return seg_begin + 1, seg_end + 1

    return seg_begin, seg_end


def get_gop_pos(frame_idx, representation):
    """given frame idx to find the group idx and the position inside a group"""
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['residual', 'mv', 'flow']:
        if gop_pos == 0:    # use the previous frame's residual and MV if it's iframe
            gop_index -= 1
            gop_pos = GOP_SIZE - 1
    else:
        gop_pos = 0 # indeed find the iframe rather than rgb frame in the middle of GOP
    return gop_index, gop_pos


class CoviarDataSet(data.Dataset):
    def __init__(self, data_root, flow_root, data_name,
                 video_list,
                 representation,
                 new_length,
                 flow_ds_factor,
                 upsample_interp,
                 transform,
                 num_segments,
                 is_train,
                 accumulate,
                 gop,
                 mv_minmaxnorm=0,
                 viz=False,
                 flow_folder='tvl1'):

        self._data_root = data_root
        self._data_name = data_name
        self._flow_root = flow_root
        self._num_segments = num_segments
        self._representation = representation
        self._new_length = new_length
        self._flow_ds_factor = flow_ds_factor
        self._upsample_interp = upsample_interp
        self._mv_minmaxnorm = mv_minmaxnorm
        self._transform = transform
        self._is_train = is_train
        self._accumulate = accumulate
        self._viz = viz
        self._flow_folder = flow_folder
        global GOP_SIZE
        GOP_SIZE = gop

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        self._load_list(video_list)

    def _load_list(self, video_list):
        self._video_list = []
        with open(video_list, 'r') as f:
            for line in f:
                video, _, label = line.strip().split()
                video_path = os.path.join(self._data_root, video[:-4] + '.mp4')
                flow_path = video_path_to_flow_path(self._flow_root, video_path)
                self._video_list.append((
                    video_path,
                    int(label),
                    min(get_num_frames(video_path),len(os.listdir(flow_path))/3)))

        print('%d videos loaded.' % len(self._video_list))

    def _get_train_frame_index(self, num_frames, seg):
        # Compute the range of the segment.
        seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg,
                                                 representation=self._representation)

        # Sample one frame from the segment.
        v_frame_idx = random.randint(seg_begin, seg_end - 1)
        return get_gop_pos(v_frame_idx, self._representation)

    def _get_test_frame_index(self, num_frames, seg):
        if self._representation in ['mv', 'residual', 'flow']:
            num_frames -= 1

        seg_size = float(num_frames - 1) / self._num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

        if self._representation in ['mv', 'residual', 'flow']:
            v_frame_idx += 1

        return get_gop_pos(v_frame_idx, self._representation)

    def __getitem__(self, index):

        if self._representation == 'mv':
            representation_idx = 1
        elif self._representation == 'residual':
            representation_idx = 2
        else:
            representation_idx = 0


        if self._is_train:
            video_path, label, num_frames = random.choice(self._video_list)
        else:
            video_path, label, num_frames = self._video_list[index]

        frames = []
        idx_first = -99999
        for seg in range(self._num_segments):

            if self._is_train:
                gop_index, gop_pos = self._get_train_frame_index(num_frames, seg)
            else:
                gop_index, gop_pos = self._get_test_frame_index(num_frames, seg)

            flow_path = video_path_to_flow_path(self._flow_root, video_path)
            if self._flow_folder == 'tvl1':
                flow_tmpl = 'flow_{0}_{1:05d}.jpg'
            if self._flow_folder[0:3] == 'PWC':
                flow_tmpl = 'flow_{0}_{1:05d}.png'
            idx = gop_index * GOP_SIZE + gop_pos + 1
            if idx_first == -99999:
                idx_first = idx
            # read the corresponding pre-computed optical flow along x and y dimension
            x_img = np.array(Image.open(os.path.join(flow_path, flow_tmpl.format('x', idx))).convert('L'))
            y_img = np.array(Image.open(os.path.join(flow_path, flow_tmpl.format('y', idx))).convert('L'))
            flow = np.stack([x_img, y_img], axis=-1)
            if flow is None:
                print('Error: loading flow %s failed.' % video_path)

            # load MV and data pre-processing
            mv = load(video_path, gop_index, gop_pos, representation_idx, self._accumulate)

            if mv is None:
                print('Error: loading video %s failed.' % video_path)
                mv = np.zeros((256, 256, 2)) if self._representation == 'mv' else np.zeros((256, 256, 3))
            else:
                if self._representation == 'mv':
                    if self._mv_minmaxnorm == 1:
                        mv = clip_and_scale(mv, 20)   # scale values from +-20 to +-127.5
                    mv += 128
                    mv = (np.minimum(np.maximum(mv, 0), 255)).astype(np.uint8)
                elif self._representation == 'residual':
                    mv += 128
                    mv = (np.minimum(np.maximum(mv, 0), 255)).astype(np.uint8)

            if self._representation == 'iframe':
                mv = color_aug(mv)

                # BGR to RGB. (PyTorch uses RGB according to doc.)
                mv = mv[..., ::-1]

            # load residual and data pre-processing
            residual = load(video_path, gop_index, gop_pos, 2, self._accumulate)
            residual += 128
            residual = (np.minimum(np.maximum(residual, 0), 255)).astype(np.uint8)

            frames.append(np.concatenate((flow, mv, residual), axis=2))

        frames = self._transform(frames)

        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))
        # print('frames shape in dataloader:')
        # print(frames.shape)  # (num_crops*num_segments, 5, 224, 224)

        # split input into input_mv and input_flow
        input_flow = frames[:, 0:2, :, :]
        input_mv = frames[:, 2:4, :, :]
        input_residual = frames[:, 4:, :, :]

        if self._flow_ds_factor is not 0:
            # downsample to make OF blocky
            factor = self._flow_ds_factor
            w_max = input_flow.shape[2]
            h_max = input_flow.shape[3]
            input_flow = block_reduce(input_flow, block_size=(1, 1, factor, factor), func=np.mean)
            # resize to original size by repeating or interpolation
            if self._upsample_interp is False:
                input_flow = input_flow.repeat(factor, axis=2).repeat(factor, axis=3)
            else:
                # interpolate along certain dimension? only interp1d can do so
                w_max_ds = input_flow.shape[2]
                h_max_ds = input_flow.shape[3]
                f_out = interpolate.interp1d(np.linspace(0, 1, w_max_ds), input_flow, kind='linear', axis=2)
                input_flow = f_out(np.linspace(0, 1, w_max_ds * factor))
                f_out = interpolate.interp1d(np.linspace(0, 1, h_max_ds), input_flow, kind='linear', axis=3)
                input_flow = f_out(np.linspace(0, 1, h_max_ds * factor))
            input_flow = input_flow[:, :, :w_max, :h_max]


        """load data from numpy to torch and pre-processing"""
        # print('input_flow shape in dataloader:')
        # print(input_flow.shape)  # (num_crops*num_segments, 2, 224, 224)
        input_flow = torch.from_numpy(input_flow).float() / 255.0
        input_mv = torch.from_numpy(input_mv).float() / 255.0
        input_residual = torch.from_numpy(input_residual).float() / 255.0
        # print('input_flow after torch shape in dataloader:')
        # print(input_flow.shape)  # torch.Size([num_crops*num_segments, 2, 224, 224])

        if self._representation == 'iframe':
            input_mv = (input_mv - self._input_mean) / self._input_std
        elif self._representation == 'mv':
            input_mv = (input_mv - 0.5) / torch.mean(self._input_std)

        input_flow = (input_flow - 0.5) / torch.mean(self._input_std)
        input_residual = (input_residual - 0.5) / self._input_std

        # print('Input flow shape %s:' % str(input_flow.shape))  # torch.Size([1, num_crops*num_segments, 2, 224, 224])
        # print('Input mv shape %s:' % str(input_mv.shape))
        # print('Input residual shape %s:' % str(input_residual.shape))
        # print('Input mv scope min %s:' % str(input_mv.min()))
        # print('Input mv scope max %s:' % str(input_mv.max()))
        # print('Input flow scope min %s:' % str(input_flow.min()))
        # print('Input flow scope max %s:' % str(input_flow.max()))
        if (self._viz == True) and (self._is_train == False):
            classname = flow_path.split('/')[-2]
            img_tmpl = 'img_{:05d}.jpg'
            # idx is the index of the first frame/segment of the current video
            return input_flow, input_mv, input_residual, label, os.path.join(flow_path, img_tmpl.format(idx_first)), classname
        else:
            return input_flow, input_mv, input_residual, label

    def __len__(self):
        return len(self._video_list)
