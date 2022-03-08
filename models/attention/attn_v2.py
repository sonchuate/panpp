from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return out
        # return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x
        # return self.sigmoid(x)


class Self_Attention_Channel(nn.Module):
    def __init__(self, inChannels, k=8):
        super(Self_Attention_Channel, self).__init__()
        # embedding_channels = inChannels // k  # C_bar

        self.key = ChannelAttention(inChannels)
        self.query = ChannelAttention(inChannels)
        self.value = ChannelAttention(inChannels)

        # self.key      = nn.Conv2d(inChannels, embedding_channels, 1)
        # self.query    = nn.Conv2d(inChannels, embedding_channels, 1)
        # self.value    = nn.Conv2d(inChannels, embedding_channels, 1)
        # self.self_att = nn.Conv2d(embedding_channels, inChannels, 1)
        self.gamma    = nn.Parameter(torch.tensor(0.0))
        self.softmax  = nn.Softmax(dim=1)

    def forward(self,t,b,m):
        """
            inputs:
                x: input feature map [Batch, Channel, Height, Width]
            returns:
                out: self attention value + input feature
                attention: [Batch, Channel, Height, Width]
        """

        ori = t + b + m

        batchsize, C, H, W = m.size()

        f_x = self.key(t).view(batchsize,   -1, C)      # Keys                  [B, C_bar, N]
        g_x = self.query(b).view(batchsize, -1, C)      # Queries               [B, C_bar, N]
        h_x = self.value(m).view(batchsize, -1, C)      # Values                [B, C_bar, N]

        s =  torch.bmm(f_x.permute(0,2,1), g_x)         # Scores                [B, N, N]
        beta = self.softmax(s)                          # Attention Map         [B, N, N]

        v = torch.bmm(h_x, beta)                        # Value x Softmax       [B, C_bar, N]
        o = v.view(batchsize, C, 1, 1)                  # Recover input shape   [B, C_bar, H, W]
        # o = self.self_att(v)                          # Self-Attention output [B, C, H, W]
        
        o = F.upsample(o, (H, W), mode='bilinear')

        y = self.gamma * o + ori                       # Learnable gamma + residual

        return y

class Self_Attention_Spatial(nn.Module):
    def __init__(self, inChannels, k=8):
        super(Self_Attention_Spatial, self).__init__()
        embedding_channels = inChannels // k  # C_bar

        self.key = SpatialAttention()
        self.query = SpatialAttention()
        self.value = SpatialAttention()

        # self.key      = nn.Conv2d(inChannels, embedding_channels, 1)
        # self.query    = nn.Conv2d(inChannels, embedding_channels, 1)
        # self.value    = nn.Conv2d(inChannels, embedding_channels, 1)
        self.self_att = nn.Conv2d(embedding_channels, inChannels, 1)
        self.gamma    = nn.Parameter(torch.tensor(0.0))
        self.softmax  = nn.Softmax(dim=1)

    def forward(self,t,b,m):
        """
            inputs:
                x: input feature map [Batch, Channel, Height, Width]
            returns:
                out: self attention value + input feature
                attention: [Batch, Channel, Height, Width]
        """

        ori = t + b + m

        batchsize, _, H, W = m.size()

        scale = 2

        t = F.adaptive_avg_pool2d(t, (H//scale, W//scale))
        b = F.adaptive_avg_pool2d(b, (H//scale, W//scale))
        m = F.adaptive_avg_pool2d(m, (H//scale, W//scale))

        N = (H//scale) * (W//scale)                                       # Number of features
        f_x = self.key(t).view(batchsize,   -1, N)      # Keys                  [B, C_bar, N]
        g_x = self.query(b).view(batchsize, -1, N)      # Queries               [B, C_bar, N]
        h_x = self.value(m).view(batchsize, -1, N)      # Values                [B, C_bar, N]

        s =  torch.bmm(f_x.permute(0,2,1), g_x)         # Scores                [B, N, N]
        beta = self.softmax(s)                          # Attention Map         [B, N, N]

        v = torch.bmm(h_x, beta)                        # Value x Softmax       [B, C_bar, N]
        v = v.view(batchsize, -1, H//scale, W//scale)   # Recover input shape   [B, C_bar, H, W]
        o = self.self_att(v)                            # Self-Attention output [B, C, H, W]
        
        o = F.upsample(o, (H, W), mode='bilinear')

        y = self.gamma * o +  ori                       # Learnable gamma + residual

        return y

class attn_v2(nn.Module):

    def __init__(self, planes):
        super(attn_v2, self).__init__()

        self.attn_channel = Self_Attention_Channel(planes,k=planes)
        self.attn_spatial = Self_Attention_Spatial(planes,k=planes)

        self.n_t = nn.Conv2d(planes,planes,1,stride=1,padding=0,bias=False)
        self.n_b = nn.Conv2d(planes,planes,1,stride=1,padding=0,bias=False)
        self.n_m = nn.Conv2d(planes,planes,1,stride=1,padding=0,bias=False)


    def forward(self, t, b, m):

        out = self.attn_channel(t,b,m)

        nt = self.n_t(out)
        nb = self.n_b(out)
        nm = self.n_m(out)

        out = self.attn_spatial(nt,nb,nm)


        # out = self.attn_spatial(t,b,m)


        return out