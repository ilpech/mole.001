import torch
from torch_dl.model import resnet2d
from torch_dl.model.resnet2dV2 import ResNetV2
import torch.nn.functional as F
from torch import nn

from varname.helpers import debug
from torch_dl.tools.tools_torch import model_params_cnt
import time

class RegressionHead(nn.Module):
    def __init__(self, feature_extractor, input_channels):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.regression_head = torch.nn.Linear(input_channels, 1)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.regression_head(x)
        return x

class RegressionResNet2d(resnet2d.ResNet2d):
    def __init__(self, block, num_blocks, inp_channels, channels_width_modifier:int=1) -> torch.nn.Module:
        super(RegressionResNet2d, self).__init__(
            block, 
            num_blocks, 
            inp_channels, 
            channels_width_modifier=channels_width_modifier
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.regression_head = torch.nn.Linear(
            64*block.expansion*channels_width_modifier, 
            1
        )
        self.bn2 = self._norm_layer(64*channels_width_modifier)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.regression_head(x)
        return x
    
    def forward_feautures_extractor(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        return x


def RegressionResNet2dV2_34(num_channels, channels_width_modifier):
    feature_extractor=ResNetV2(
        num_channels,
        [2, 2, 4],
        channels_width_modifier,
        out_with_features=False
    )
    sample = torch.zeros(size=(1,num_channels, 100, 100))
    out = feature_extractor(sample)
    return RegressionHead(
        feature_extractor=feature_extractor,
        input_channels=out.shape[1]
    )
    
def RegressionResNet2dV2_50(num_channels, channels_width_modifier):
    feature_extractor=ResNetV2(
        num_channels,
        [3, 4, 9],
        channels_width_modifier,
        out_with_features=False
    )
    sample = torch.zeros(size=(1,num_channels, 100, 100))
    out = feature_extractor(sample)
    return RegressionHead(
        feature_extractor=feature_extractor,
        input_channels=out.shape[1]
    )


def RegressionResNet2d18(num_channels, channels_width_modifier):
    return RegressionResNet2d(resnet2d.BasicBlock, [2, 2, 2, 2], inp_channels=num_channels, channels_width_modifier=channels_width_modifier)

def RegressionResNet2d34(num_channels, channels_width_modifier):
    return RegressionResNet2d(resnet2d.BasicBlock, [3, 4, 6, 3], inp_channels=num_channels, channels_width_modifier=channels_width_modifier)

def RegressionResNet2d50(num_channels, channels_width_modifier):
    return RegressionResNet2d(resnet2d.Bottleneck, [3, 4, 6, 3], inp_channels=num_channels, channels_width_modifier=channels_width_modifier)

def RegressionResNet2d101(num_channels, channels_width_modifier):
    return RegressionResNet2d(resnet2d.Bottleneck, [3, 4, 23, 3], inp_channels=num_channels, channels_width_modifier=channels_width_modifier)

def RegressionResNet2d152(num_channels, channels_width_modifier):
    return RegressionResNet2d(resnet2d.Bottleneck, [3, 8, 36, 3], inp_channels=num_channels, channels_width_modifier=channels_width_modifier)
    
if __name__ == '__main__':
    from torch_dl.tools.tools_torch import inspect_model
    sample = torch.randn((1,10,2048,21))
    model = RegressionResNet2d34(
        num_channels=sample.shape[1], 
        channels_width_modifier=3
    )
    inspect_model(model, sample, 'RegressionResNet2d34_3')
    model = RegressionResNet2dV2_34(
        num_channels=sample.shape[1], 
        channels_width_modifier=1
    )
    inspect_model(model, sample, 'RegressionResNet2dV2_34')
    model = RegressionResNet2dV2_50(
        num_channels=sample.shape[1], 
        channels_width_modifier=1
    )
    inspect_model(model, sample, 'RegressionResNet2dV2_50')
    # one_sample_ts = []
    # for i in range(100):
    #     one_sample_inf = time.time()
    #     out = model(sample)
    #     # one_sample_ts.append()
    # debug(out.shape)
    # debug(model_params_cnt(model))
