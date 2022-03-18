from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import Conv_BN_ReLU


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
        self.sigmoid = nn.Sigmoid()

        self.part1_chnls = int(inChannels * part_ratio)


    def forward_csp(self,q,k,v):
        """
            inputs:
                x: input feature map [Batch, Channel, Height, Width]
            returns:
                out: self attention value + input feature
                attention: [Batch, Channel, Height, Width]
        """

        ###### CSP

        part1_q = q[:, :self.part1_chnls, :, :]
        part1_k = k[:, :self.part1_chnls, :, :]
        part1_v = v[:, :self.part1_chnls, :, :]

        part2_q = q[:, self.part1_chnls:, :, :]
        part2_k = q[:, self.part1_chnls:, :, :]
        part2_v = q[:, self.part1_chnls:, :, :]

        ori = part1_q + part1_k + part1_v

        ######
        batchsize, C, H, W = part2_q.size()

        f_x = self.key(part2_k).view(batchsize,   -1, C)      # Keys                  [B, C_bar, N]
        g_x = self.query(part2_q).view(batchsize, -1, C)      # Queries               [B, C_bar, N]
        h_x = part2_v.view(batchsize, -1, C)                      # Values                [B, C_bar, N]

        s =  torch.bmm(f_x.permute(0,2,1), g_x)         # Scores                [B, N, N]
        beta = self.softmax(s)                          # Attention Map         [B, N, N]

        part2_v = torch.bmm(h_x, beta)                        # Value x Softmax       [B, C_bar, N]
        o = part2_v.view(batchsize, C, H, W)                  # Recover input shape   [B, C_bar, H, W]

        o = self.sigmoid(o)

        y = o * ori        

        out = torch.cat((ori, y), 1)     
        
        return out

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


class attn_v4(nn.Module):

    def __init__(self, planes):
        super(attn_v4, self).__init__()

        k = 4

        self.t_1 = Channel_Attention(planes,k=k)
        self.t_2 = Channel_Attention(planes,k=k)
        self.t_3 = Channel_Attention(planes,k=k)

        self.b_1 = Channel_Attention(planes,k=k)
        self.b_2 = Channel_Attention(planes,k=k)
        self.b_3 = Channel_Attention(planes,k=k)

        self.m_1 = Channel_Attention(planes,k=k)
        self.m_2 = Channel_Attention(planes,k=k)
        self.m_3 = Channel_Attention(planes,k=k)

        # Conv 1x1
        self.t_q = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0)
        self.t_k = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0)
        self.t_v = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0)

        self.b_q = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0)
        self.b_k = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0)
        self.b_v = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0)

        self.m_q = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0)
        self.m_k = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0)
        self.m_v = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0)

        self.out_t = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0)
        self.out_b = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0)
        self.out_m = Conv_BN_ReLU(planes,planes,1,stride=1,padding=0)

    def forward(self, t, b, m):

        t_q, t_k, t_v = self.t_q(t), self.t_k(t), self.t_v(t)
        b_q, b_k, b_v = self.b_q(b), self.b_k(b), self.b_v(b)
        m_q, m_k, m_v = self.m_q(m), self.m_k(m), self.m_v(m)

        out_t = self.t_1(t_q, t_k, t_v) + self.t_2(b_q, t_k, b_v) + self.t_3(m_q, t_k, m_v)
        out_b = self.b_1(t_q, b_k, t_v) + self.b_2(b_q, b_k, b_v) + self.b_3(m_q, b_k, m_v)
        out_m = self.m_1(t_q, m_k, t_v) + self.m_2(b_q, m_k, b_v) + self.m_3(m_q, m_k, m_v)

        out = self.out_t(out_t) + self.out_b(out_b) + self.out_m(out_m)

        return out