import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .padding import pad_same


def conv2d_same(
    x,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """
    一个自定义的2D卷积层类，旨在处理输入张量以保持其尺寸在经过卷积后不变。
    
    该类通过特定的padding策略确保了卷积操作后输出的高和宽与输入相同。
    
    参数:
    - in_channels (int): 输入张量的通道数。
    - out_channels (int): 输出张量的通道数。
    - kernel_size (int 或 tuple): 卷积核的大小。
    - stride (int 或 tuple, 可选): 卷积的步长，默认为1。
    - padding (int 或 str, 可选): 对输入进行padding的策略，默认为0，表示无padding。
    - dilation (int 或 tuple, 可选): 卷积核元素之间的间距，默认为1。
    - groups (int, 可选): 输入通道上分组卷积的个数，默认为1，表示不分组。
    - bias (bool, 可选): 如果为True，则在卷积层后添加一个偏置项，默认为True。
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2dSame, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
            dilation,
            groups,
            bias,
        )

    def forward(self, x):
        return conv2d_same(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
