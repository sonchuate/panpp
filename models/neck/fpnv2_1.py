import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from ..utils import Conv_BN_ReLU

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


def _middle_cat(u, d, m=None):

    if m == None: # only up, down
        _, _, u_H, u_W = u.size()
        _, _, d_H, d_W = d.size()

        m_H = (u_H + d_H) // 2
        m_W = (u_W + d_W) // 2

        u = F.adaptive_avg_pool2d(u, (m_H, m_W))
        d = F.upsample(d, (m_H, m_W), mode='bilinear')

        f = torch.cat((u, d), 1)

    else: # up, down, mid
        _, _, m_H, m_W = m.size()

        u = F.adaptive_avg_pool2d(u, (m_H, m_W))
        d = F.upsample(d, (m_H, m_W), mode='bilinear')

        f = torch.cat((u, d, m), 1)

    return f


def _adaptive_cat(x, y): # upsamle y to x's size

    _, _, H, W = x.size()
    y = F.upsample(y, size=(H, W), mode='bilinear')

    return torch.cat((x, y), 1)


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Conv_block, self).__init__()

        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.smooth_layer = Conv_BN_ReLU(in_channels, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):

        x = self.dwconv(x)
        x = self.smooth_layer(x)
        x = self.conv(x)

        return x

class repeat_block(nn.Module):
    def __init__(self, out_channels):
        super(repeat_block, self).__init__()
        planes = out_channels

        self.L1 = Conv_block(in_channels=planes*2, out_channels=planes, stride=1)
        self.L2 = Conv_block(in_channels=planes*3, out_channels=planes, stride=1)
        self.L3 = Conv_block(in_channels=planes*3, out_channels=planes, stride=1)
        self.L4 = Conv_block(in_channels=planes*2, out_channels=planes, stride=1)

        self.M1 = Conv_block(in_channels=planes*3, out_channels=planes, stride=1)
        self.M2 = Conv_block(in_channels=planes*3, out_channels=planes, stride=1)
        self.M3 = Conv_block(in_channels=planes*3, out_channels=planes, stride=1)

        self.F1 = Conv_block(in_channels=planes*2, out_channels=planes, stride=1)
        self.F2 = Conv_block(in_channels=planes*3, out_channels=planes, stride=1)
        self.F3 = Conv_block(in_channels=planes*3, out_channels=planes, stride=1)
        self.F4 = Conv_block(in_channels=planes*2, out_channels=planes, stride=1)

    def forward(self, features):

        f1, E1, f2, E2, f3, E3, f4 = features

        L1 = self.L1(_adaptive_cat(f1,E1))
        L2 = self.L2(_middle_cat(E1,E2,m=f2))
        L3 = self.L3(_middle_cat(E2,E3,m=f3))
        L4 = self.L4(_adaptive_cat(f4,E3))

        M1 = self.M1(_middle_cat(L1,L2,m=E1))
        M2 = self.M2(_middle_cat(L2,L3,m=E2))
        M3 = self.M3(_middle_cat(L3,L4,m=E3))

        F1 = self.F1(_adaptive_cat(L1,M1))
        F2 = self.F2(_middle_cat(M1,M2,m=L2))
        F3 = self.F3(_middle_cat(M2,M3,m=L3))
        F4 = self.F4(_adaptive_cat(L4,M3))

        return F1, M1, F2, M2, F3, M3, F4

class FPN_v2_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN_v2_1, self).__init__()
        planes = out_channels

        self.E1 = Conv_block(in_channels=planes*2, out_channels=planes, stride=1)
        self.E2 = Conv_block(in_channels=planes*2, out_channels=planes, stride=1)
        self.E3 = Conv_block(in_channels=planes*2, out_channels=planes, stride=1)

        self.R = 1
        self.fpn_layers = []
        for i in range(0, self.R):
            self.fpn_layers.append(repeat_block(out_channels=planes))
        self.fpn = nn.Sequential(*self.fpn_layers)

        self.last_conv = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)


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
