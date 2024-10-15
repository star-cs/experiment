from torch import nn
from .layers.blocks import *
import torch.nn.functional as F
import torch
from timm.models.layers import trunc_normal_

def UpsamplingNearest2d(x, scale_factor=2, mode='nearest'):
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)

class DUCK_Net(nn.Module):
    def __init__(self, in_channels):
        super(DUCK_Net, self).__init__()
        
        # 展示这么设计，后期还可以把换成下五层。
        # 这个层数有点低。
        self.channel_list = [6, 12, 24, 48]
        self.feature_size = [128, 64, 32, 16]

        self.conv1 = Conv2dSamePadding(3, in_channels * 2, kernel_size=2, stride=2)
        self.conv2 = Conv2dSamePadding(
            in_channels * 2, in_channels * 4, kernel_size=2, stride=2
        )
        self.conv3 = Conv2dSamePadding(
            in_channels * 4, in_channels * 8, kernel_size=2, stride=2
        )
        self.conv4 = Conv2dSamePadding(
            in_channels * 8, in_channels * 16, kernel_size=2, stride=2
        )

        self.t0 = Conv_Block(3, in_channels, "duckv2")

        self.l1i = Conv2dSamePadding(
            in_channels, in_channels * 2, kernel_size=2, stride=2
        )
        self.l2i = Conv2dSamePadding(
            in_channels * 2, in_channels * 4, kernel_size=2, stride=2
        )
        self.l3i = Conv2dSamePadding(
            in_channels * 4, in_channels * 8, kernel_size=2, stride=2
        )
        self.l4i = Conv2dSamePadding(
            in_channels * 8, in_channels * 16, kernel_size=2, stride=2
        )

        self.t1 = Conv_Block(in_channels * 2, in_channels * 2, "duckv2")
        self.t2 = Conv_Block(in_channels * 4, in_channels * 4, "duckv2")
        self.t3 = Conv_Block(in_channels * 8, in_channels * 8, "duckv2")
        self.t4 = Conv_Block(in_channels * 16, in_channels * 16, "duckv2")  

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            fan_in // m.groups
            std = math.sqrt(2.0 / fan_in)
            m.weight.data.normal_(0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # 修改 DUCK_Net 网络，
        # t_0 层不输出，因为没有什么信息
        # 输出 t_1 ~ t_4 
        # 不输出 t_5，因为第五层没有 DUCK block 操作。
        output = []
        p1 = self.conv1(x)
        p2 = self.conv2(p1)
        p3 = self.conv3(p2)
        p4 = self.conv4(p3)

        t_0 = self.t0(x)  # 第一层
        l1_i = self.l1i(t_0) # 卷积 向下走
        s_1 = p1 + l1_i  # 
        t_1 = self.t1(s_1)  # 每一层最后一个conv，t_1就是每一层的输出层。
        output.append(t_1)

        l2_i = self.l2i(t_1)
        s_2 = p2 + l2_i
        t_2 = self.t2(s_2)
        output.append(t_2)

        l3_i = self.l3i(t_2)
        s_3 = p3 + l3_i
        t_3 = self.t3(s_3)
        output.append(t_3)

        l4_i = self.l4i(t_3)
        s_4 = p4 + l4_i
        t_4 = self.t4(s_4)
        output.append(t_4)

        return output
    
if __name__ == "__main__":
    model = DUCK_Net(in_channels=3)
    model.eval()
    x = torch.randn(1,3,256,256)
    y = model(x)
    #torch.save(model.state_dict(), 'cat.pth')
    for i in y:
        print(i.shape)