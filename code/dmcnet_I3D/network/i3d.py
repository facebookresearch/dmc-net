"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import os

import numpy as np
import torch
from torch.nn import ReplicationPad3d
from torch import nn
try:
    from . import initializer
except:
    import initializer
from torch.utils.checkpoint import checkpoint_sequential
"""
Define the network architecture of MV->MV_SD generator
The following type of EstimatorDenseNet uses dense connections
"""
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))    
def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


class EstimatorDenseNet(nn.Module):
    def __init__(self, ch_in):
        super(EstimatorDenseNet, self).__init__()
        self.conv_0 = conv(ch_in,       128, kernel_size=3, stride=1)
        dd = 128
        self.conv_1 = conv(ch_in+dd, 128, kernel_size=3, stride=1)
        dd += 128
        self.conv_2 = conv(ch_in+dd, 96,  kernel_size=3, stride=1)
        dd += 96
        self.conv_3 = conv(ch_in+dd, 64,  kernel_size=3, stride=1)
        dd += 64
        self.conv_4 = conv(ch_in+dd, 32,  kernel_size=3, stride=1)
        dd += 32
        self.predict_flow = predict_flow(ch_in+dd)

    def forward(self, x):
        # input mv+residual 5x224x224 output flow or delta 2x224x224
        x = torch.cat((self.conv_0(x), x), 1)
        x = torch.cat((self.conv_1(x), x), 1)
        x = torch.cat((self.conv_2(x), x), 1)
        x = torch.cat((self.conv_3(x), x), 1)
        x = torch.cat((self.conv_4(x), x), 1)
        return self.predict_flow(x)


class EstimatorDenseNetSmall(nn.Module):
    def __init__(self, ch_in):
        super(EstimatorDenseNetSmall, self).__init__()
        self.conv_0 = conv(ch_in,       32, kernel_size=3, stride=1)
        dd = 32
        self.conv_1 = conv(ch_in+dd, 32, kernel_size=3, stride=1)
        dd += 32
        self.conv_2 = conv(ch_in+dd, 24,  kernel_size=3, stride=1)
        dd += 24
        self.conv_3 = conv(ch_in+dd, 16,  kernel_size=3, stride=1)
        dd += 16
        self.conv_4 = conv(ch_in+dd, 8,  kernel_size=3, stride=1)
        dd += 8
        self.predict_flow = predict_flow(ch_in+dd)


    def forward(self, x):
        # input mv+residual 5x224x224 output flow or delta 2x224x224
        x = torch.cat((self.conv_0(x), x), 1)
        x = torch.cat((self.conv_1(x), x), 1)
        x = torch.cat((self.conv_2(x), x), 1)
        x = torch.cat((self.conv_3(x), x), 1)
        x = torch.cat((self.conv_4(x), x), 1)
        return self.predict_flow(x)



class EstimatorDenseNetTiny(nn.Module):
    def __init__(self, ch_in):
        super(EstimatorDenseNetTiny, self).__init__()
        self.conv_0 = conv(ch_in,       8, kernel_size=3, stride=1)
        dd = 8
        self.conv_1 = conv(ch_in+dd, 8, kernel_size=3, stride=1)
        dd += 8
        self.conv_2 = conv(ch_in+dd, 6,  kernel_size=3, stride=1)
        dd += 6
        self.conv_3 = conv(ch_in+dd, 4,  kernel_size=3, stride=1)
        dd += 4
        self.conv_4 = conv(ch_in+dd, 2,  kernel_size=3, stride=1)
        dd += 2
        self.predict_flow = predict_flow(ch_in+dd)

    def forward(self, x):
        # input mv+residual 5x224x224 output flow or delta 2x224x224
        x = torch.cat((self.conv_0(x), x), 1)
        x = torch.cat((self.conv_1(x), x), 1)
        x = torch.cat((self.conv_2(x), x), 1)
        x = torch.cat((self.conv_3(x), x), 1)
        x = torch.cat((self.conv_4(x), x), 1)
        return self.predict_flow(x)

"""
Define the network architecture of discriminator in GAN
Start with unit blocks and then define several classes
"""
def discriminator_block(in_filters, out_filters, bn=True):
    block = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25))
    if bn:
        block = nn.Sequential(
                    nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25),
                    nn.BatchNorm2d(out_filters, 0.8))
    return block


def discriminator_block2(in_filters, out_filters, bn=True):
    block = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25))
    if bn:
        block = nn.Sequential(
                    nn.Conv2d(in_filters, out_filters, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25),
                    nn.BatchNorm2d(out_filters, 0.8))
    return block


class Discriminator(nn.Module):
    def __init__(self, ch_in):
        super(Discriminator, self).__init__()

        self.discriminator_block_1 = discriminator_block(ch_in, 16, bn=False)
        self.discriminator_block_2 = discriminator_block(16, 32)
        self.discriminator_block_3 = discriminator_block(32, 64)
        self.discriminator_block_4 = discriminator_block(64, 128)
        self.adv_layer = nn.Linear(128*14*14, 2)

    def forward(self, x):
        model = nn.Sequential(self.discriminator_block_1,
        self.discriminator_block_2,
        self.discriminator_block_3,
        self.discriminator_block_4)
        x = model(x)
        x = x.view(x.shape[0], -1)
        validity = self.adv_layer(x)

        return validity

class Discriminator2(nn.Module):
    def __init__(self, ch_in):
        super(Discriminator2, self).__init__()

        self.discriminator_block_1 = discriminator_block(ch_in, 16, bn=False)
        self.discriminator_block_1_2 = discriminator_block2(16, 16)
        self.discriminator_block_2 = discriminator_block(16, 32)
        self.discriminator_block_2_2 = discriminator_block2(32, 32)
        self.discriminator_block_3 = discriminator_block(32, 64)
        self.discriminator_block_3_2 = discriminator_block2(64, 64)
        self.discriminator_block_4 = discriminator_block(64, 128)
        self.discriminator_block_4_2 = discriminator_block2(128, 128)
        self.adv_layer = nn.Linear(128*14*14, 2)

    def forward(self, x):
        x = self.discriminator_block_1(x)
        x = self.discriminator_block_1_2(x)
        x = self.discriminator_block_2(x)
        x = self.discriminator_block_2_2(x)
        x = self.discriminator_block_3(x)
        x = self.discriminator_block_3_2(x)
        x = self.discriminator_block_4(x)
        x = self.discriminator_block_4_2(x)
        x = x.view(x.shape[0], -1)
        validity = self.adv_layer(x)

        return validity


class Discriminator3(nn.Module):
    def __init__(self, ch_in):
        super(Discriminator3, self).__init__()

        self.discriminator_block_1 = discriminator_block(ch_in, 16, bn=False)
        self.discriminator_block_1_2 = discriminator_block2(16, 16)
        self.discriminator_block_1_3 = discriminator_block2(16, 16)
        self.discriminator_block_2 = discriminator_block(16, 32)
        self.discriminator_block_2_2 = discriminator_block2(32, 32)
        self.discriminator_block_2_3 = discriminator_block2(32, 32)
        self.discriminator_block_3 = discriminator_block(32, 64)
        self.discriminator_block_3_2 = discriminator_block2(64, 64)
        self.discriminator_block_3_3 = discriminator_block2(64, 64)
        self.discriminator_block_4 = discriminator_block(64, 128)
        self.discriminator_block_4_2 = discriminator_block2(128, 128)
        self.discriminator_block_4_3 = discriminator_block2(128, 128)
        self.adv_layer = nn.Linear(128*14*14, 2)

    def forward(self, x):
        x = self.discriminator_block_1(x)
        x = self.discriminator_block_1_2(x)
        x = self.discriminator_block_1_3(x)
        x = self.discriminator_block_2(x)
        x = self.discriminator_block_2_2(x)
        x = self.discriminator_block_2_3(x)
        x = self.discriminator_block_3(x)
        x = self.discriminator_block_3_2(x)
        x = self.discriminator_block_3_3(x)
        x = self.discriminator_block_4(x)
        x = self.discriminator_block_4_2(x)
        x = self.discriminator_block_4_3(x)
        x = x.view(x.shape[0], -1)
        validity = self.adv_layer(x)

        return validity


class Discriminator4(nn.Module):
    def __init__(self, ch_in):
        super(Discriminator4, self).__init__()

        self.discriminator_block_1 = discriminator_block(ch_in, 8, bn=False)
        self.discriminator_block_2 = discriminator_block(8, 16)
        self.discriminator_block_3 = discriminator_block(16, 32)
        self.adv_layer = nn.Linear(32*28*28, 2)

    def forward(self, x):
        x = self.discriminator_block_1(x)
        x = self.discriminator_block_2(x)
        x = self.discriminator_block_3(x)
        x = x.view(x.shape[0], -1)
        validity = self.adv_layer(x)

        return validity


class Discriminator5(nn.Module):
    def __init__(self, ch_in):
        super(Discriminator5, self).__init__()

        self.discriminator_block_1 = discriminator_block(ch_in, 16, bn=False)
        self.discriminator_block_1_2 = discriminator_block2(16, 16)
        self.discriminator_block_1_3 = discriminator_block2(16, 16)
        self.discriminator_block_1_4 = discriminator_block2(16, 16)
        self.discriminator_block_1_5 = discriminator_block2(16, 16)
        self.discriminator_block_2 = discriminator_block(16, 32)
        self.discriminator_block_2_2 = discriminator_block2(32, 32)
        self.discriminator_block_2_3 = discriminator_block2(32, 32)
        self.discriminator_block_2_4 = discriminator_block2(32, 32)
        self.discriminator_block_2_5 = discriminator_block2(32, 32)
        self.discriminator_block_3 = discriminator_block(32, 64)
        self.discriminator_block_3_2 = discriminator_block2(64, 64)
        self.discriminator_block_3_3 = discriminator_block2(64, 64)
        self.discriminator_block_3_4 = discriminator_block2(64, 64)
        self.discriminator_block_3_5 = discriminator_block2(64, 64)
        self.discriminator_block_4 = discriminator_block(64, 128)
        self.discriminator_block_4_2 = discriminator_block2(128, 128)
        self.discriminator_block_4_3 = discriminator_block2(128, 128)
        self.discriminator_block_4_4 = discriminator_block2(128, 128)
        self.discriminator_block_4_5 = discriminator_block2(128, 128)
        self.adv_layer = nn.Linear(128*14*14, 2)

    def forward(self, x):
        x = self.discriminator_block_1(x)
        x = self.discriminator_block_1_2(x)
        x = self.discriminator_block_1_3(x)
        x = self.discriminator_block_1_4(x)
        x = self.discriminator_block_1_5(x)
        x = self.discriminator_block_2(x)
        x = self.discriminator_block_2_2(x)
        x = self.discriminator_block_2_3(x)
        x = self.discriminator_block_2_4(x)
        x = self.discriminator_block_2_5(x)
        x = self.discriminator_block_3(x)
        x = self.discriminator_block_3_2(x)
        x = self.discriminator_block_3_3(x)
        x = self.discriminator_block_3_4(x)
        x = self.discriminator_block_3_5(x)
        x = self.discriminator_block_4(x)
        x = self.discriminator_block_4_2(x)
        x = self.discriminator_block_4_3(x)
        x = self.discriminator_block_4_4(x)
        x = self.discriminator_block_4_5(x)
        x = x.view(x.shape[0], -1)
        validity = self.adv_layer(x)

        return validity
    
def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init


class Unit3Dpy(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation='relu',
                 padding='SAME',
                 use_bias=False,
                 use_bn=True, squeeze = False, mean = False):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        self.squeeze = squeeze
        self.mean = mean
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == 'VALID':
            padding_shape = 0
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if padding == 'SAME':
            if not simplify_pad:
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=use_bias)
            else:
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=pad_size,
                    bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding_shape,
                stride=stride,
                bias=use_bias)
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if self.use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)

        if activation == 'relu':
            self.activation = torch.nn.functional.relu

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = torch.nn.functional.relu(out)
        if self.squeeze:
            out = out.squeeze(3)
            out = out.squeeze(3)
            if self.mean:
                out = out.mean(2)
        return out


class MaxPool3dTFPadding(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class Mixed(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mixed, self).__init__()
        # Branch 0
        self.branch_0 = Unit3Dpy(
            in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Unit3Dpy(
            in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3Dpy(
            out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = torch.nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3Dpy(
            in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3Dpy(
            out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = torch.nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Unit3Dpy(
            in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = torch.nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out


class I3D(torch.nn.Module):
    def __init__(self,
                 num_classes,
                 modality='rgb',
                 dropout_prob=0,
                 arch_estimator = None,
                 arch_d = None,
                 name='inception', **kwargs):
        super(I3D, self).__init__()

        self.name = name
        self.num_classes = num_classes
        if modality == 'rgb':
            in_channels = 3
        elif modality in ['flow', 'mv', 'flow+mp4']:
            in_channels = 2
        elif modality == 'res' or 'I':
            in_channels = 3
        else:
            raise ValueError(
                '{} not among known modalities [rgb|flow]'.format(modality))
        self.modality = modality
        
        #build the generator part
        self.arch_estimator = arch_estimator
        if self.arch_estimator == 'DenseNet':
            self.gen_flow_model = EstimatorDenseNet(5)
        if self.arch_estimator == 'DenseNetSmall':
            self.gen_flow_model = EstimatorDenseNetSmall(5)
        if self.arch_estimator == 'DenseNetTiny':
            self.gen_flow_model = EstimatorDenseNetTiny(5)
        self.arch_d = arch_d   
        # setup the discriminator in GAN
        if self.arch_d == 'Discriminator':
            self.discriminator = Discriminator(2)
        elif self.arch_d == 'Discriminator2':
            self.discriminator = Discriminator2(2)
        elif self.arch_d == 'Discriminator3':
            self.discriminator = Discriminator3(2)
        elif self.arch_d == 'Discriminator4':
            self.discriminator = Discriminator4(2)
        elif self.arch_d == 'Discriminator5':
            self.discriminator = Discriminator5(2)

        conv3d_1a_7x7 = Unit3Dpy(
            out_channels=64,
            in_channels=in_channels,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding='SAME')
        # 1st conv-pool
        self.conv3d_1a_7x7 = conv3d_1a_7x7
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        # conv conv
        conv3d_2b_1x1 = Unit3Dpy(
            out_channels=64,
            in_channels=64,
            kernel_size=(1, 1, 1),
            padding='SAME')
        self.conv3d_2b_1x1 = conv3d_2b_1x1
        conv3d_2c_3x3 = Unit3Dpy(
            out_channels=192,
            in_channels=64,
            kernel_size=(3, 3, 3),
            padding='SAME')
        self.conv3d_2c_3x3 = conv3d_2c_3x3
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')

        # Mixed_3b
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])

        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')

        # Mixed 4
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])

        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')

        # Mixed 5
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024,
            #out_channels=self.num_classes,
            out_channels = 400,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False, squeeze = True, mean = True)
        self.classifier = torch.nn.Linear(400, num_classes)
        self.softmax = torch.nn.Softmax(1)


    def forward(self, inp, node = 'logit', detach = False):
        if node == 'D':
            return self.discriminator(inp)
                                     
        #generating flow part
        if self.arch_estimator in ['DenseNet','DenseNetSmall', 'DenseNetTiny']:
            b, c, t, h, w = inp.shape
            inp = self.gen_flow_model(torch.reshape(torch.transpose(inp, 1, 2), (-1, c, h, w)))
            inp = torch.transpose(torch.reshape(inp, (b, t, 2, h, w)), 1, 2)
        # Preprocessing
        if detach:
            out = self.conv3d_1a_7x7(inp.detach())
        else:
            out = self.conv3d_1a_7x7(inp)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxPool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        out = self.avg_pool(out)
        out = self.conv3d_0c_1x1(out)
        out = self.dropout(out)
        out = self.classifier(out)
        if node == 'flow+logit':
            return out, inp
        elif node == 'gen_flow':
            return inp
        else:
            return out
        
        

    def load_tf_weights(self, sess):
        state_dict = {}
        if self.modality == 'rgb':
            prefix = 'RGB/inception_i3d'
        elif self.modality == 'flow':
            prefix = 'Flow/inception_i3d'
        load_conv3d(state_dict, 'conv3d_1a_7x7', sess,
                    os.path.join(prefix, 'Conv3d_1a_7x7'))
        load_conv3d(state_dict, 'conv3d_2b_1x1', sess,
                    os.path.join(prefix, 'Conv3d_2b_1x1'))
        load_conv3d(state_dict, 'conv3d_2c_3x3', sess,
                    os.path.join(prefix, 'Conv3d_2c_3x3'))

        load_mixed(state_dict, 'mixed_3b', sess,
                   os.path.join(prefix, 'Mixed_3b'))
        load_mixed(state_dict, 'mixed_3c', sess,
                   os.path.join(prefix, 'Mixed_3c'))
        load_mixed(state_dict, 'mixed_4b', sess,
                   os.path.join(prefix, 'Mixed_4b'))
        load_mixed(state_dict, 'mixed_4c', sess,
                   os.path.join(prefix, 'Mixed_4c'))
        load_mixed(state_dict, 'mixed_4d', sess,
                   os.path.join(prefix, 'Mixed_4d'))
        load_mixed(state_dict, 'mixed_4e', sess,
                   os.path.join(prefix, 'Mixed_4e'))
        # Here goest to 0.1 max error with tf
        load_mixed(state_dict, 'mixed_4f', sess,
                   os.path.join(prefix, 'Mixed_4f'))

        load_mixed(
            state_dict,
            'mixed_5b',
            sess,
            os.path.join(prefix, 'Mixed_5b'),
            fix_typo=True)
        load_mixed(state_dict, 'mixed_5c', sess,
                   os.path.join(prefix, 'Mixed_5c'))
        load_conv3d(
            state_dict,
            'conv3d_0c_1x1',
            sess,
            os.path.join(prefix, 'Logits', 'Conv3d_0c_1x1'),
            bias=True,
            bn=False)
        self.load_state_dict(state_dict)


def get_conv_params(sess, name, bias=False):
    # Get conv weights
    conv_weights_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'w:0'))
    if bias:
        conv_bias_tensor = sess.graph.get_tensor_by_name(
            os.path.join(name, 'b:0'))
        conv_bias = sess.run(conv_bias_tensor)
    conv_weights = sess.run(conv_weights_tensor)
    conv_shape = conv_weights.shape

    kernel_shape = conv_shape[0:3]
    in_channels = conv_shape[3]
    out_channels = conv_shape[4]

    conv_op = sess.graph.get_operation_by_name(
        os.path.join(name, 'convolution'))
    padding_name = conv_op.get_attr('padding')
    padding = _get_padding(padding_name, kernel_shape)
    all_strides = conv_op.get_attr('strides')
    strides = all_strides[1:4]
    conv_params = [
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding
    ]
    if bias:
        conv_params.append(conv_bias)
    return conv_params


def get_bn_params(sess, name):
    moving_mean_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_mean:0'))
    moving_var_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_variance:0'))
    beta_tensor = sess.graph.get_tensor_by_name(os.path.join(name, 'beta:0'))
    moving_mean = sess.run(moving_mean_tensor)
    moving_var = sess.run(moving_var_tensor)
    beta = sess.run(beta_tensor)
    return moving_mean, moving_var, beta


def _get_padding(padding_name, conv_shape):
    padding_name = padding_name.decode("utf-8")
    if padding_name == "VALID":
        return [0, 0]
    elif padding_name == "SAME":
        #return [math.ceil(int(conv_shape[0])/2), math.ceil(int(conv_shape[1])/2)]
        return [
            math.floor(int(conv_shape[0]) / 2),
            math.floor(int(conv_shape[1]) / 2),
            math.floor(int(conv_shape[2]) / 2)
        ]
    else:
        raise ValueError('Invalid padding name ' + padding_name)


def load_conv3d(state_dict, name_pt, sess, name_tf, bias=False, bn=True):
    # Transfer convolution params
    conv_name_tf = os.path.join(name_tf, 'conv_3d')
    conv_params = get_conv_params(sess, conv_name_tf, bias=bias)
    if bias:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding, conv_bias = conv_params
    else:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding = conv_params

    conv_weights_rs = np.transpose(
        conv_weights, (4, 3, 0, 1,
                       2))  # to pt format (out_c, in_c, depth, height, width)
    state_dict[name_pt + '.conv3d.weight'] = torch.from_numpy(conv_weights_rs)
    if bias:
        state_dict[name_pt + '.conv3d.bias'] = torch.from_numpy(conv_bias)

    # Transfer batch norm params
    if bn:
        conv_tf_name = os.path.join(name_tf, 'batch_norm')
        moving_mean, moving_var, beta = get_bn_params(sess, conv_tf_name)

        out_planes = conv_weights_rs.shape[0]
        state_dict[name_pt + '.batch3d.weight'] = torch.ones(out_planes)
        state_dict[name_pt + '.batch3d.bias'] = torch.from_numpy(beta)
        state_dict[name_pt
                   + '.batch3d.running_mean'] = torch.from_numpy(moving_mean)
        state_dict[name_pt
                   + '.batch3d.running_var'] = torch.from_numpy(moving_var)


def load_mixed(state_dict, name_pt, sess, name_tf, fix_typo=False):
    # Branch 0
    load_conv3d(state_dict, name_pt + '.branch_0', sess,
                os.path.join(name_tf, 'Branch_0/Conv3d_0a_1x1'))

    # Branch .1
    load_conv3d(state_dict, name_pt + '.branch_1.0', sess,
                os.path.join(name_tf, 'Branch_1/Conv3d_0a_1x1'))
    load_conv3d(state_dict, name_pt + '.branch_1.1', sess,
                os.path.join(name_tf, 'Branch_1/Conv3d_0b_3x3'))

    # Branch 2
    load_conv3d(state_dict, name_pt + '.branch_2.0', sess,
                os.path.join(name_tf, 'Branch_2/Conv3d_0a_1x1'))
    if fix_typo:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess,
                    os.path.join(name_tf, 'Branch_2/Conv3d_0a_3x3'))
    else:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess,
                    os.path.join(name_tf, 'Branch_2/Conv3d_0b_3x3'))

    # Branch 3
    load_conv3d(state_dict, name_pt + '.branch_3.1', sess,
    os.path.join(name_tf, 'Branch_3/Conv3d_0b_1x1'))
    
    
