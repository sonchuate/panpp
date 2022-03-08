from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
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

        self.key = ChannelAttention(embedding_channels)
        self.query = ChannelAttention(embedding_channels)
        self.value = ChannelAttention(embedding_channels)

        self.reduce_channel = nn.Conv2d(inChannels,embedding_channels,1,stride=1,padding=0,bias=False)
        self.recovery_channel = nn.Conv2d(embedding_channels,inChannels,1,stride=1,padding=0,bias=False)

        self.softmax  = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self,q,k,v):
        """
            inputs:
                x: input feature map [Batch, Channel, Height, Width]
            returns:
                out: self attention value + input feature
                attention: [Batch, Channel, Height, Width]
        """

        ori = q + k + v

        q, k, v = self.reduce_channel(q), self.reduce_channel(k), self.reduce_channel(v)

        q, k, v = self.relu(q), self.relu(k), self.relu(v)

        batchsize, C, H, W = q.size()

        f_x = self.key(k).view(batchsize,   -1, C)      # Keys                  [B, C_bar, N]
        g_x = self.query(q).view(batchsize, -1, C)      # Queries               [B, C_bar, N]
        h_x = self.value(v).view(batchsize, -1, C)      # Values                [B, C_bar, N]

        s =  torch.bmm(f_x.permute(0,2,1), g_x)         # Scores                [B, N, N]
        beta = self.softmax(s)                          # Attention Map         [B, N, N]

        v = torch.bmm(h_x, beta)                        # Value x Softmax       [B, C_bar, N]
        o = v.view(batchsize, C, 1, 1)                  # Recover input shape   [B, C_bar, H, W]

        o = self.recovery_channel(o)

        o = F.upsample(o, (H, W), mode='bilinear')

        o = self.sigmoid(o)

        y = o * ori                     

        return y

# class Self_Attention_Channel(nn.Module):
#     def __init__(self, inChannels, k=8):
#         super(Self_Attention_Channel, self).__init__()
#         # embedding_channels = inChannels // k  # C_bar

#         self.key = ChannelAttention(inChannels)
#         self.query = ChannelAttention(inChannels)
#         self.value = ChannelAttention(inChannels)

#         # self.key      = nn.Conv2d(inChannels, embedding_channels, 1)
#         # self.query    = nn.Conv2d(inChannels, embedding_channels, 1)
#         # self.value    = nn.Conv2d(inChannels, embedding_channels, 1)
#         # self.self_att = nn.Conv2d(embedding_channels, inChannels, 1)
#         self.gamma    = nn.Parameter(torch.tensor(0.0))
#         self.softmax  = nn.Softmax(dim=1)

#     def forward(self,t,b,m):
#         """
#             inputs:
#                 x: input feature map [Batch, Channel, Height, Width]
#             returns:
#                 out: self attention value + input feature
#                 attention: [Batch, Channel, Height, Width]
#         """

#         ori = t + b + m

#         batchsize, C, H, W = m.size()

#         f_x = self.key(t).view(batchsize,   -1, C)      # Keys                  [B, C_bar, N]
#         g_x = self.query(b).view(batchsize, -1, C)      # Queries               [B, C_bar, N]
#         h_x = self.value(m).view(batchsize, -1, C)      # Values                [B, C_bar, N]

#         s =  torch.bmm(f_x.permute(0,2,1), g_x)         # Scores                [B, N, N]
#         beta = self.softmax(s)                          # Attention Map         [B, N, N]

#         v = torch.bmm(h_x, beta)                        # Value x Softmax       [B, C_bar, N]
#         o = v.view(batchsize, C, 1, 1)                  # Recover input shape   [B, C_bar, H, W]
#         # o = self.self_att(v)                          # Self-Attention output [B, C, H, W]
        
#         o = F.upsample(o, (H, W), mode='bilinear')

#         y = self.gamma * o + ori                       # Learnable gamma + residual

#         return y

# class Self_Attention_Spatial(nn.Module):
#     def __init__(self, inChannels, k=8):
#         super(Self_Attention_Spatial, self).__init__()
#         embedding_channels = inChannels // k  # C_bar

#         self.key = SpatialAttention()
#         self.query = SpatialAttention()
#         self.value = SpatialAttention()

#         # self.key      = nn.Conv2d(inChannels, embedding_channels, 1)
#         # self.query    = nn.Conv2d(inChannels, embedding_channels, 1)
#         # self.value    = nn.Conv2d(inChannels, embedding_channels, 1)
#         self.self_att = nn.Conv2d(embedding_channels, inChannels, 1)
#         self.gamma    = nn.Parameter(torch.tensor(0.0))
#         self.softmax  = nn.Softmax(dim=1)

#     def forward(self,t,b,m):
#         """
#             inputs:
#                 x: input feature map [Batch, Channel, Height, Width]
#             returns:
#                 out: self attention value + input feature
#                 attention: [Batch, Channel, Height, Width]
#         """

#         ori = t + b + m

#         batchsize, _, H, W = m.size()

#         scale = 2

#         t = F.adaptive_avg_pool2d(t, (H//scale, W//scale))
#         b = F.adaptive_avg_pool2d(b, (H//scale, W//scale))
#         m = F.adaptive_avg_pool2d(m, (H//scale, W//scale))

#         N = (H//scale) * (W//scale)                                       # Number of features
#         f_x = self.key(t).view(batchsize,   -1, N)      # Keys                  [B, C_bar, N]
#         g_x = self.query(b).view(batchsize, -1, N)      # Queries               [B, C_bar, N]
#         h_x = self.value(m).view(batchsize, -1, N)      # Values                [B, C_bar, N]

#         s =  torch.bmm(f_x.permute(0,2,1), g_x)         # Scores                [B, N, N]
#         beta = self.softmax(s)                          # Attention Map         [B, N, N]

#         v = torch.bmm(h_x, beta)                        # Value x Softmax       [B, C_bar, N]
#         v = v.view(batchsize, -1, H//scale, W//scale)   # Recover input shape   [B, C_bar, H, W]
#         o = self.self_att(v)                            # Self-Attention output [B, C, H, W]
        
#         o = F.upsample(o, (H, W), mode='bilinear')

#         y = self.gamma * o +  ori                       # Learnable gamma + residual

#         return y

class attn_v3(nn.Module):

    def __init__(self, planes):
        super(attn_v3, self).__init__()

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
        self.t_q = nn.Conv2d(planes,planes,1,stride=1,padding=0,bias=False)
        self.t_k = nn.Conv2d(planes,planes,1,stride=1,padding=0,bias=False)
        self.t_v = nn.Conv2d(planes,planes,1,stride=1,padding=0,bias=False)

        self.b_q = nn.Conv2d(planes,planes,1,stride=1,padding=0,bias=False)
        self.b_k = nn.Conv2d(planes,planes,1,stride=1,padding=0,bias=False)
        self.b_v = nn.Conv2d(planes,planes,1,stride=1,padding=0,bias=False)

        self.m_q = nn.Conv2d(planes,planes,1,stride=1,padding=0,bias=False)
        self.m_k = nn.Conv2d(planes,planes,1,stride=1,padding=0,bias=False)
        self.m_v = nn.Conv2d(planes,planes,1,stride=1,padding=0,bias=False)

    def forward(self, t, b, m):

        t_q, t_k, t_v = self.t_q(t), self.t_k(t), self.t_v(t)
        b_q, b_k, b_v = self.b_q(b), self.b_k(b), self.b_v(b)
        m_q, m_k, m_v = self.m_q(m), self.m_k(m), self.m_v(m)

        out_t = self.t_1(t_q, t_k, t_v) + self.t_2(b_q, t_k, b_v) + self.t_3(m_q, t_k, m_v)
        out_b = self.b_1(t_q, b_k, t_v) + self.b_2(b_q, b_k, b_v) + self.b_3(m_q, b_k, m_v)
        out_m = self.m_1(t_q, m_k, t_v) + self.m_2(b_q, m_k, b_v) + self.m_3(m_q, m_k, m_v)

        out = out_t + out_b + out_m

        return out