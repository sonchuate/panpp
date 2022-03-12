from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..attention import attn_v4 as attn

# from ..utils import Conv_BN_ReLU

def _middle_cat(u, d):

    _, _, u_H, u_W = u.size()
    _, _, d_H, d_W = d.size()

    m_H = (u_H + d_H) // 2
    m_W = (u_W + d_W) // 2

    u = F.adaptive_avg_pool2d(u, (m_H, m_W))
    d = F.upsample(d, (m_H, m_W), mode='bilinear')

    f = torch.cat((u, d), 1)

    return f


class Attention_Module(nn.Module): # upsamle y to x's size

    def __init__(self,input_channel):
        super(Attention_Module, self).__init__()

        self.attn = attn(input_channel)

    def forward(self, top, bot, mid):

        _, _, m_H, m_W = mid.size()

        top = F.adaptive_avg_pool2d(top, (m_H, m_W))
        bot = F.upsample(bot, (m_H, m_W), mode='bilinear')

        out = self.attn(top, bot, mid)

        return out

class Conv_BN_ReLU(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Conv_block, self).__init__()

        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.smooth_layer = Conv_BN_ReLU(in_channels, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.dwconv(x)
        x = self.smooth_layer(x)
        x = self.conv(x)
        x = self.bn(x)
        # x = self.relu(x)

        return x

class repeat_block(nn.Module):
    def __init__(self, out_channels):
        super(repeat_block, self).__init__()
        planes = out_channels

        self.L1 = Conv_block(in_channels=planes, out_channels=planes, stride=1)
        self.L2 = Conv_block(in_channels=planes, out_channels=planes, stride=1)
        self.L3 = Conv_block(in_channels=planes, out_channels=planes, stride=1)
        self.L4 = Conv_block(in_channels=planes, out_channels=planes, stride=1)

        self.M1 = Conv_block(in_channels=planes, out_channels=planes, stride=1)
        self.M2 = Conv_block(in_channels=planes, out_channels=planes, stride=1)
        self.M3 = Conv_block(in_channels=planes, out_channels=planes, stride=1)

        self.F1 = Conv_block(in_channels=planes, out_channels=planes, stride=1)
        self.F2 = Conv_block(in_channels=planes, out_channels=planes, stride=1)
        self.F3 = Conv_block(in_channels=planes, out_channels=planes, stride=1)
        self.F4 = Conv_block(in_channels=planes, out_channels=planes, stride=1)

        self.L1_attn_module = Attention_Module(planes)
        self.L2_attn_module = Attention_Module(planes)
        self.L3_attn_module = Attention_Module(planes)
        self.L4_attn_module = Attention_Module(planes)

        self.M1_attn_module = Attention_Module(planes)
        self.M2_attn_module = Attention_Module(planes)
        self.M3_attn_module = Attention_Module(planes)

        self.F1_attn_module = Attention_Module(planes)
        self.F2_attn_module = Attention_Module(planes)
        self.F3_attn_module = Attention_Module(planes)
        self.F4_attn_module = Attention_Module(planes)

    def forward(self, features):

        f1, E1, f2, E2, f3, E3, f4 = features

        L1 = self.L1(self.L1_attn_module(f1,E1,f1)) # top bot mid
        L2 = self.L2(self.L2_attn_module(E1,E2,f2))
        L3 = self.L3(self.L3_attn_module(E2,E3,f3))
        L4 = self.L4(self.L4_attn_module(E3,f4,f4))

        M1 = self.M1(self.M1_attn_module(L1,L2,E1))
        M2 = self.M2(self.M2_attn_module(L2,L3,E2))
        M3 = self.M3(self.M3_attn_module(L3,L4,E3))

        # F1 = self.F1(self.F1_attn_module(L1,M1,L1))
        F1 = self.F1(self.F1_attn_module(f1,M1,L1))
        F2 = self.F2(self.F2_attn_module(M1,M2,L2))
        F3 = self.F3(self.F3_attn_module(M2,M3,L3))
        # F4 = self.F4(self.F4_attn_module(M3,L4,L4))
        F4 = self.F4(self.F4_attn_module(M3,f4,L4))

        return F1, M1, F2, M2, F3, M3, F4

class FPN_v3_3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FPN_v3_3, self).__init__()

        planes = out_channels

        self.E1 = Conv_block(in_channels=planes*2, out_channels=planes, stride=1)
        self.E2 = Conv_block(in_channels=planes*2, out_channels=planes, stride=1)
        self.E3 = Conv_block(in_channels=planes*2, out_channels=planes, stride=1)

        self.R = 1
        self.fpn_layers = []
        for _ in range(0, self.R):
            self.fpn_layers.append(repeat_block(out_channels=planes))
        self.fpn = nn.Sequential(*self.fpn_layers)

        self.last_conv = Conv_BN_ReLU(planes, planes)



    def forward(self, f1, f2, f3, f4):

        # expand layer
        E1 = self.E1(_middle_cat(f1,f2))
        E2 = self.E2(_middle_cat(f2,f3))
        E3 = self.E3(_middle_cat(f3,f4))

        features = f1, E1, f2, E2, f3, E3, f4

        # repeat fpn block
        features = self.fpn(features)

        # conv 1x1
        for f in features:
            f = self.last_conv(f)

        F1, M1, F2, M2, F3, M3, F4 = features

        return F1, M1, F2, M2, F3, M3, F4
