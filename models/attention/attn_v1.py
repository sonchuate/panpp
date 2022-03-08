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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim // 1, kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels =  in_dim // 1, kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,t,b,m):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        ori = t + b + m
        _, _, width_ori, height_ori = ori.size()

        scale = 2

        t = F.adaptive_avg_pool2d(t, (width_ori//scale, height_ori//scale))
        b = F.adaptive_avg_pool2d(b, (width_ori//scale, height_ori//scale))
        m = F.adaptive_avg_pool2d(m, (width_ori//scale, height_ori//scale))

        m_batchsize, C, width, height = m.size()

        proj_query  = self.query_conv(t).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(b).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(m).view(m_batchsize,-1,width*height) # B X C X N
        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)

        out = F.upsample(out, (width_ori, height_ori), mode='bilinear')
        
        out = out + ori

        return out

class Self_Attention(nn.Module):
    def __init__(self, inChannels, k=8):
        super(Self_Attention, self).__init__()
        embedding_channels = inChannels // k  # C_bar
        self.key      = nn.Conv2d(inChannels, embedding_channels, 1)
        self.query    = nn.Conv2d(inChannels, embedding_channels, 1)
        self.value    = nn.Conv2d(inChannels, embedding_channels, 1)
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

class attn_v1(nn.Module):

    def __init__(self, planes):
        super(attn_v1, self).__init__()

        # self.attn = Self_Attn(planes)
        self.attn = Self_Attention(planes,k=planes)

    def forward(self, t, b, m):

        out = self.attn(t,b,m)


        return out