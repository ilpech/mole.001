import math

from os.path import join as pjoin
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_dl.model.std_conv import StdConv2d

from varname.helpers import debug, varname

def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

class ResNetV2(nn.Module):
    """Pre-activation (v2) ResNet mode."""
    def __init__(
        self, 
        input_channels:int,
        block_units: List[int], 
        width_factor:int,
        out_with_features=False
    ):
        super().__init__()
        self.input_channels = input_channels
        self.block_units = block_units
        self.width_factor = width_factor
        self.out_with_features = out_with_features
        width = int(64 * self.width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            (
                'conv', 
                StdConv2d(
                    input_channels, 
                    width, 
                    kernel_size=7, 
                    stride=2, 
                    bias=False, 
                    padding=3
                )
            ),
            ('gn', nn.GroupNorm(
                32, 
                width, 
                eps=1e-6
            )),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(
                    cin=width, 
                    cout=width*4, 
                    cmid=width
                ))] +
                [(f'unit{i:d}', PreActBottleneck(
                    cin=width*4, 
                    cout=width*4, 
                    cmid=width
                )) for i in range(2, self.block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(
                    cin=width*4, 
                    cout=width*8, 
                    cmid=width*2, 
                    stride=2
                ))] +
                [(f'unit{i:d}', PreActBottleneck(
                    cin=width*8, 
                    cout=width*8, 
                    cmid=width*2
                )) for i in range(2, self.block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(
                    cin=width*8, 
                    cout=width*16, 
                    cmid=width*4, 
                    stride=2
                ))] +
                [(f'unit{i:d}', PreActBottleneck(
                    cin=width*16, 
                    cout=width*16, 
                    cmid=width*4
                )) for i in range(2, self.block_units[2] + 1)],
            ))),
        ]))

    def forward(self, x):
        b, c, in_size, _ = x.size()
        x = self.root(x)
        if self.out_with_features:
            features = []
            features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            if self.out_with_features:
                right_size = int(in_size / 4 / (i+1))
                if x.size()[2] != right_size:
                    pad = right_size - x.size()[2]
                    assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                    feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                    feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
                else:
                    feat = x
                features.append(feat)
        x = self.body[-1](x)
        if self.out_with_features:
            return x, features[::-1]
        return x
    

    

if __name__ == '__main__':
    from torch_dl.tools.tools_torch import inspect_model
    sample_batch = torch.zeros(size=(1,10,2048,21))
    resnet34_1 = ResNetV2(
        input_channels=10,
        block_units=[2,2,4],
        width_factor=1,
        out_with_features=True
    )
    inspect_model(
        resnet34_1,
        sample_batch,
        'resnet34_1'
    )
    resnet34_2 = ResNetV2(
        input_channels=10,
        block_units=[2,2,4],
        width_factor=2
    )
    inspect_model(
        resnet34_2,
        sample_batch,
        'resnet34_2'
    )
    resnet50_1 = ResNetV2(
        input_channels=10,
        block_units=[3,4,9],
        width_factor=1
    )
    inspect_model(
        resnet50_1,
        sample_batch,
        'resnet50_1'
    )
    resnet50_2 = ResNetV2(
        input_channels=10,
        block_units=[3,4,9],
        width_factor=2
    )
    inspect_model(
        resnet50_2,
        sample_batch,
        'resnet50_2'
    )
    