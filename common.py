# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn

from typing import Type
import torch.nn.init as init


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
    
class Reshaper(nn.Module):
    
    def __init__(self, size_in: int, channel_in: int, size_out: int, channel_out: int,
                 normalization: Type[nn.Module] = LayerNorm2d,
                 activation: Type[nn.Module] = nn.GELU):
        """
        初始化Reshaper类的实例。

        该构造函数根据输入的图像尺寸和通道数以及期望的输出尺寸和通道数，
        构建一个神经网络模块列表，用于在图像特征图之间进行形状转换。
        支持尺寸增加（上采样）、尺寸减少（下采样）和尺寸不变的情况。
        使用反卷积（上卷积）进行上采样，用卷积进行下采样。

        参数:
        - size_in: 输入特征图的尺寸。
        - channel_in: 输入特征图的通道数。
        - size_out: 期望输出特征图的尺寸。
        - channel_out: 输出特征图的通道数。
        - normalization: 用于特征图的归一化层类型，默认为LayerNorm2d。
        - activation: 用于特征图的激活函数类型，默认为nn.GELU。

        抛出:
        - ValueError: 如果size_out不是size_in的倍数，或者size_in不是size_out的倍数（取决于情况）。
        """
        
        super(Reshaper, self).__init__()
        
        # 当输入和输出尺寸相同时，只需一个1x1卷积调整通道数
        if size_in == size_out:
            self.reshaper = nn.ModuleList([
                nn.Conv2d(channel_in, channel_out, kernel_size=1),
            ])
            
        # 当输入尺寸小于输出尺寸时，使用反卷积（上卷积）进行上采样
        elif size_in < size_out:
            n = math.log2(size_out // size_in)
            if n.is_integer():
                n = int(n)
            else:
                raise ValueError(f"size_out must be a multiple of size_in, got {size_out} and {size_in}")
            
            self.reshaper = nn.ModuleList()
            for _ in range(n):
                self.reshaper.extend([
                    nn.ConvTranspose2d(channel_in, channel_in // 2, kernel_size=2, stride=2),
                    normalization(channel_in // 2),
                    activation(),
                ])
                channel_in = channel_in // 2
            self.reshaper.extend([
                nn.Conv2d(channel_in, channel_out, kernel_size=1),
            ])
        
        # 当输入尺寸大于输出尺寸时，使用卷积进行下采样
        else:
            n = math.log2(size_in // size_out)
            if n.is_integer():
                n = int(n)
            else:
                raise ValueError(f"size_in must be a multiple of size_out, got {size_in} and {size_out}")
            
            self.reshaper = nn.ModuleList()
            for _ in range(n):
                self.reshaper.extend([
                    nn.Conv2d(channel_in, channel_in * 2, kernel_size=3, stride=2, padding=1),
                    normalization(channel_in * 2),
                    activation(),
                ])
                channel_in = channel_in * 2
            self.reshaper.extend([
                nn.Conv2d(channel_in, channel_out, kernel_size=1),
            ])
        self._initialize_weights()
            
    
    def _initialize_weights(self):
        for module in self.reshaper:
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.fill_(0)
    
    def forward(self, x):
        for layer in self.reshaper:
            x = layer(x)
        return x


def dice_coeff(pred, label):
    smooth = 1.
    bs = pred.size(0)
    m1 = pred.contiguous().view(bs, -1)
    m2 = label.contiguous().view(bs, -1)
    intersection = (m1 * m2).sum()
    score = 1 - ((2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth))
    return score

def bce_loss(pred, label):
    score = torch.nn.BCELoss()(pred, label)
    return score