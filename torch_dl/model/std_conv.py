import torch
import torch.nn as nn
import torch.nn.functional as F


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(
            w, 
            dim=[1, 2, 3], 
            keepdim=True, 
            unbiased=False
        )
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(
            x, 
            w, 
            self.bias, 
            self.stride, 
            self.padding,
            self.dilation, 
            self.groups
        )