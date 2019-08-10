"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import numpy as np
from skimage.measure import block_reduce
from scipy import interpolate
from .image_transforms import Compose, \
                              Transform, \
                              Normalize, \
                              Resize, \
                              RandomScale, \
                              CenterCrop, \
                              RandomCrop, \
                              RandomHorizontalFlip, \
                              RandomRGB, \
                              RandomHLS


class ToTensor(Transform):
    """Converts a numpy.ndarray (H x W x (T x C)) in the range
    [0, 255] to a torch.FloatTensor of shape (C x T x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, modality = 'rgb', flow_ds_factor = 1, interp = False):
        self.modality = modality
        self._flow_ds_factor = flow_ds_factor
        self._upsample_interp = interp
        if modality == 'rgb':
            self.dim = 3
        elif modality in ['flow', 'mv']:
            self.dim = 2
        elif modality in ['res', 'I']:
            self.dim = 3
        elif modality == 'flow+mp4':
            self.dim = 7

    def __call__(self, clips):
        if isinstance(clips, np.ndarray):
            H, W, _ = clips.shape
            # handle numpy array
            clips = clips.reshape((H,W,-1,self.dim)).transpose((3, 2, 0, 1))
            if self.modality == 'flow+mp4':
                if self._flow_ds_factor is not 0 or 1:
                    clips = np.transpose(clips, (1,0,2,3))
                    # downsample to make OF blocky
                    factor = self._flow_ds_factor
                    w_max = H
                    h_max = W
                    input_flow = block_reduce(clips[:,0:2, :, :], block_size=(1, 1, factor, factor), func=np.mean)
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
                    clips[:,0:2, :, :] = input_flow[:, :, :w_max, :h_max]
                clips = np.transpose(clips, (1,0,2,3))
                
            clips = torch.from_numpy(clips)
            #print(clips.shape)
            # backward compatibility
            return clips.float() / 255.0