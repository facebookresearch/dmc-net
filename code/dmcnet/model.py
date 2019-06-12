"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""Model definition."""

import torch
from torch import nn
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
import torchvision

import logging
logging.basicConfig(level=logging.DEBUG)

"""
Define the network architecture of MV->MV_SD generator
The following type of ContextNetwork uses dilation conv
"""
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def conv_dilation(batch_norm, in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


class ContextNetwork(nn.Module):
    def __init__(self, ch_in, batch_norm=True, gen_flow_ds_factor=0):
        super(ContextNetwork, self).__init__()
        if gen_flow_ds_factor == 0:
            self.conv_context = nn.Sequential(
                conv_dilation(batch_norm, ch_in, 32, 3, 1, 1),
                conv_dilation(batch_norm, 32, 128, 3, 1, 2),
                conv_dilation(batch_norm, 128, 128, 3, 1, 4),
                conv_dilation(batch_norm, 128, 96, 3, 1, 8),
                conv_dilation(batch_norm, 96, 64, 3, 1, 16),
                conv_dilation(batch_norm, 64, 32, 3, 1, 1),
                conv_dilation(batch_norm, 32, 2, 3, 1, 1)
            )
        else:
            self.conv_context = nn.Sequential(
                conv_dilation(batch_norm, ch_in, 32, 3, 1, 1),
                conv_dilation(batch_norm, 32, 128, 3, 1, 2),
                conv_dilation(batch_norm, 128, 128, 3, 1, 4),
                conv_dilation(batch_norm, 128, 96, 3, 1, 8),
                conv_dilation(batch_norm, 96, 64, 3, 1, 1),
                conv_dilation(batch_norm, 64, 32, 3, 1, 1),
                conv_dilation(batch_norm, 32, 2, 3, 1, 1)
            )

    def forward(self, x):
        # input 5x224x224 output 2x224x224
        return self.conv_context(x)


class ContextNetworkAtt(nn.Module):
    def __init__(self, ch_in, batch_norm=True, gen_flow_ds_factor=0):
        super(ContextNetworkAtt, self).__init__()
        if gen_flow_ds_factor == 0:
            self.conv_context = nn.Sequential(
                conv_dilation(batch_norm, ch_in, 32, 3, 1, 1),
                conv_dilation(batch_norm, 32, 128, 3, 1, 2),
                conv_dilation(batch_norm, 128, 128, 3, 1, 4),
                conv_dilation(batch_norm, 128, 96, 3, 1, 8),
                conv_dilation(batch_norm, 96, 64, 3, 1, 16),
                conv_dilation(batch_norm, 64, 32, 3, 1, 1)
            )
        else:
            self.conv_context = nn.Sequential(
                conv_dilation(batch_norm, ch_in, 32, 3, 1, 1),
                conv_dilation(batch_norm, 32, 128, 3, 1, 2),
                conv_dilation(batch_norm, 128, 128, 3, 1, 4),
                conv_dilation(batch_norm, 128, 96, 3, 1, 8),
                conv_dilation(batch_norm, 96, 64, 3, 1, 1),
                conv_dilation(batch_norm, 64, 32, 3, 1, 1)
            )
        self.predict_flow = conv_dilation(batch_norm, 32, 2, 3, 1, 1)
        self.predict_att = nn.Sequential(
                conv_dilation(batch_norm, 32, 2, 3, 1, 1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        # input 5x224x224 output 2x224x224
        x = self.conv_context(x)
        return self.predict_flow(x), self.predict_att(x)


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


class EstimatorDenseNetTinyEarlyFusionSum(nn.Module):
    def __init__(self, ch_in):
        super(EstimatorDenseNetTinyEarlyFusionSum, self).__init__()
        self.conv_0_mv = conv(2,       8, kernel_size=3, stride=1)
        self.conv_0_r = conv(3,       8, kernel_size=3, stride=1)
        dd = 8
        self.conv_1 = conv(dd, 8, kernel_size=3, stride=1)
        dd += 8
        self.conv_2 = conv(dd, 6,  kernel_size=3, stride=1)
        dd += 6
        self.conv_3 = conv(dd, 4,  kernel_size=3, stride=1)
        dd += 4
        self.conv_4 = conv(dd, 2,  kernel_size=3, stride=1)
        dd += 2
        self.predict_flow = predict_flow(dd)

    def forward(self, x):
        # input mv+residual 5x224x224 output flow or delta 2x224x224
        x_mv = self.conv_0_mv(x[:, :2,:,:])
        x_r = self.conv_0_r(x[:, 2:,:,:])
        x = x_mv+x_r
        x = torch.cat((self.conv_1(x), x), 1)
        x = torch.cat((self.conv_2(x), x), 1)
        x = torch.cat((self.conv_3(x), x), 1)
        x = torch.cat((self.conv_4(x), x), 1)
        return self.predict_flow(x)


class EstimatorDenseNetTinyEarlyFusionStack(nn.Module):
    def __init__(self, ch_in):
        super(EstimatorDenseNetTinyEarlyFusionStack, self).__init__()
        self.conv_0_mv = conv(2,       8, kernel_size=3, stride=1)
        self.conv_0_r = conv(3,       8, kernel_size=3, stride=1)
        dd = 16
        self.conv_1 = conv(dd, 8, kernel_size=3, stride=1)
        dd += 8
        self.conv_2 = conv(dd, 6,  kernel_size=3, stride=1)
        dd += 6
        self.conv_3 = conv(dd, 4,  kernel_size=3, stride=1)
        dd += 4
        self.conv_4 = conv(dd, 2,  kernel_size=3, stride=1)
        dd += 2
        self.predict_flow = predict_flow(dd)

    def forward(self, x):
        # input mv+residual 5x224x224 output flow or delta 2x224x224
        x_mv = self.conv_0_mv(x[:, :2,:,:])
        x_r = self.conv_0_r(x[:, 2:,:,:])
        x = torch.cat((x_mv, x_r), 1)
        x = torch.cat((self.conv_1(x), x), 1)
        x = torch.cat((self.conv_2(x), x), 1)
        x = torch.cat((self.conv_3(x), x), 1)
        x = torch.cat((self.conv_4(x), x), 1)
        return self.predict_flow(x)

"""define the whole model"""
class Model(nn.Module):
    def __init__(self, num_class, num_segments, representation,
                 base_model='resnet152', new_length=1, use_databn=1, gen_flow_or_delta=0, gen_flow_ds_factor=0, arch_estimator='ContextNetwork', att=0):
        super(Model, self).__init__()
        self._representation = representation
        self.num_segments = num_segments
        self.new_length = new_length
        self.use_databn = use_databn
        self.gen_flow_or_delta = gen_flow_or_delta
        self.gen_flow_ds_factor = gen_flow_ds_factor
        self.arch_estimator = arch_estimator
        self.att = att

        print(("""
Initializing model:
    base model:         {}.
    input_representation:     {}.
    num_class:          {}.
    num_segments:       {}.
    new_length:       {}.
        """.format(base_model, self._representation, num_class, self.num_segments, self.new_length)))

        # setup network architecture
        # part 1: the classification backbone network accepting MV_SD as input
        # part 2: the generator of MV->MV_SD
        self._prepare_base_model(base_model)
        # setup the TSN framework
        self._prepare_tsn(num_class)

    # setup the TSN framework
    def _prepare_tsn(self, num_class):

        feature_dim = getattr(self.base_model, 'fc').in_features
        setattr(self.base_model, 'fc', nn.Linear(feature_dim, num_class))

        if self._representation == 'mv' or self._representation == 'flow':
            setattr(self.base_model, 'conv1',
                    nn.Conv2d(2 * self.new_length, 64,
                              kernel_size=(7, 7),
                              stride=(2, 2),
                              padding=(3, 3),
                              bias=False))
            if self.use_databn == 1:
                self.data_bn = nn.BatchNorm2d(2)
        if self._representation == 'residual':
            if self.use_databn == 1:
                self.data_bn = nn.BatchNorm2d(3)

    def _prepare_base_model(self, base_model):

        # setup the classification network accepting MV_SD as input
        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(pretrained=True)
            self._input_size = 224
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

        # setup the generator network of MV->MV_SD
        if self.arch_estimator is 'ContextNetwork':
            if self.att == 0:
                self.gen_flow_model = ContextNetwork(5, True, self.gen_flow_ds_factor)
            if self.att == 1:
                self.gen_flow_model = ContextNetworkAtt(5, True, self.gen_flow_ds_factor)
        if self.arch_estimator == 'DenseNet':
            self.gen_flow_model = EstimatorDenseNet(5)
        if self.arch_estimator == 'DenseNetSmall':
            self.gen_flow_model = EstimatorDenseNetSmall(5)
        if self.arch_estimator == 'DenseNetTiny':
            self.gen_flow_model = EstimatorDenseNetTiny(5)
        if self.arch_estimator == 'DenseNetTinyEarlyFusionSum':
            self.gen_flow_model = EstimatorDenseNetTinyEarlyFusionSum(5)
        if self.arch_estimator == 'DenseNetTinyEarlyFusionStack':
            self.gen_flow_model = EstimatorDenseNetTinyEarlyFusionStack(5)
        if self.gen_flow_ds_factor is not 0:
            self.downsample = nn.AvgPool2d(self.gen_flow_ds_factor, stride=self.gen_flow_ds_factor)

    # define the forward pass of the whole model
    def forward(self, input_mv, input_residual):

        # prepare input consisting of MV and residual for generating MV_SD
        input_mv = input_mv.view((-1, ) + input_mv.size()[-3:])
        input_residual = input_residual.view((-1, ) + input_residual.size()[-3:])
        if self.gen_flow_ds_factor is not 0:
            input_mv = self.downsample(input_mv)
            input_residual = self.downsample(input_residual)

        # generate MV_SD
        if self.att == 0:
            gen_flow = self.gen_flow_model(torch.cat((input_mv, input_residual), 1))
        elif self.att == 1:
            gen_flow, att_flow = self.gen_flow_model(torch.cat((input_mv, input_residual), 1))

        if self.gen_flow_or_delta == 1: # generate flow delta
            gen_flow = torch.add(gen_flow, input_mv)
        if self.gen_flow_ds_factor is not 0:
            gen_flow = gen_flow.repeat(1, 1, self.gen_flow_ds_factor, self.gen_flow_ds_factor)
            # gen_flow = torch.nn.functional.upsample(gen_flow, scale_factor=self.gen_flow_ds_factor, mode='bilinear')

        # Feed MV_SD for classification
        base_out = self.base_model(gen_flow.detach())

        if self.att == 0:
            return base_out, gen_flow
        elif self.att == 1:
            return base_out, gen_flow, att_flow

    @property
    def crop_size(self):
        #logging.debug('self.crop_size: {}'.format(self._input_size))
        return self._input_size

    @property
    def scale_size(self):
        #logging.debug('self.scale_size: {}'.format(self._input_size * 256 // 224)) # called during testing. all input images are scaled to 224*256//224=256
        return self._input_size * 256 // 224

    def get_augmentation(self):
        if self._representation in ['mv', 'residual', 'flow']:
            scales = [1, .875, .75]
        else:
            scales = [1, .875, .75, .66]

        print('Augmentation scales:', scales)
        return torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales),
             GroupRandomHorizontalFlip()])
