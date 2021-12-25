# Copyright (c) Runpei Dong, ArChip Lab.

""" DGMS convolution implementation.

Author: Runpei Dong
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg
from .GMM import *

class DGMSConv(nn.Conv2d):
    """ DGMS Convolution: 
    Convolution operator based on Differentiable Gaussian Mixture Weight Sharing (DGMS) for model compression.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode: str = 'zeros',
    ):
        super(DGMSConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.is_normal = cfg.IS_NORMAL

        self.k_level = cfg.K_LEVEL
        self.temperature = cfg.TAU

    def init_mask_params(self):
        init_method = 'empirical' if cfg.IS_EMP else 'k-means'
        self.sub_distribution = gmm_approximation(self.k_level, self.weight, self.temperature, init_method)

    def get_Sweight(self):
        # soft quantized weights during training
        with torch.no_grad():
            return self.sub_distribution(weights=self.weight, train=True)

    def get_Pweight(self):
        # hard quantized weights during inference
        with torch.no_grad():
            return self.sub_distribution(weights=self.weight, train=False) 

    def forward(self, input):
        if cfg.IS_NORMAL:
            # pretraning using normal convolution operator
            output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            # DGMS convolution operator
            if cfg.IS_TRAIN:
                # training using DGMS differentiable indicator
                Sweight = self.sub_distribution(weights=self.weight, train=True)
                output = F.conv2d(input, Sweight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                # inference using hard mask
                Pweight = self.sub_distribution(weights=self.weight, train=False)
                output = F.conv2d(input, Pweight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


# unit test script
if __name__ == '__main__':
    m = DGMSConv(16, 33, 3, stride=2)
    input = torch.randn(20, 16, 50, 100)
    output = m(input)
    print(output.size())
