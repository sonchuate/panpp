from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# from ..utils import Conv_BN_ReLU

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3., self.inplace) / 6.
        return out * x


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
        # self.relu = h_swish()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, pool_size, ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(pool_size)
        self.max_pool = nn.AdaptiveMaxPool2d(pool_size)
           
        # self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
        #                        nn.ReLU(),
        #                        nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
  
        return self.avg_pool(x) + self.max_pool(x)

    # def forward(self, x):
    #     avg_out = self.fc(self.avg_pool(x))
    #     max_out = self.fc(self.max_pool(x))
    #     out = avg_out + max_out
    #     return out
    #     # return self.sigmoid(out)

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()

#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
#         # self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return x
#         # return self.sigmoid(x)


class Channel_Attention(nn.Module):
    def __init__(self, inChannels, k):
        super(Channel_Attention, self).__init__()
        embedding_channels = inChannels // k  # C_bar

        pool_size = 1
        part_ratio = 0.5 # keep ratio

        self.key = ChannelAttention(embedding_channels, pool_size=pool_size)
        self.query = ChannelAttention(embedding_channels, pool_size=pool_size)


        self.softmax  = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()
        self.sigmoid = h_sigmoid()

        self.part1_chnls = int(inChannels * part_ratio)

    def forward(self,q,k,v):
        """
            inputs:
                x: input feature map [Batch, Channel, Height, Width]
            returns:
                out: self attention value + input feature
                attention: [Batch, Channel, Height, Width]
        """

        ori = q + k + v

        batchsize, C, H, W = q.size()

        f_x = self.key(k).view(batchsize,   -1, C)      # Keys                  [B, C_bar, N]
        g_x = self.query(q).view(batchsize, -1, C)      # Queries               [B, C_bar, N]
        h_x = v.view(batchsize, -1, C)                      # Values                [B, C_bar, N]

        s =  torch.bmm(f_x.permute(0,2,1), g_x)         # Scores                [B, N, N]
        beta = self.softmax(s)                          # Attention Map         [B, N, N]

        v = torch.bmm(h_x, beta)                        # Value x Softmax       [B, C_bar, N]
        o = v.view(batchsize, C, H, W)                  # Recover input shape   [B, C_bar, H, W]

        
        o = self.sigmoid(o)

        y = o * ori                     
        
        return y

### original
class attn_v5_2(nn.Module):

    def __init__(self, planes):
        super(attn_v5_2, self).__init__()

        k = 4

        self.t = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0)
        self.b = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0)
        self.m = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0)

        self.q = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0) 
        self.k = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0) 
        self.v = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0) 

        self.attn = Channel_Attention(planes,k=k)

    def forward(self, t, b, m):

        mix_fusion = self.t(t) + self.b(b) + self.m(m)

        q, k, v = self.q(mix_fusion), self.k(mix_fusion), self.v(mix_fusion)

        out = self.attn(q, k, v)

        return out