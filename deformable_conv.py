import torch
from torchvision.ops import DeformConv2d
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d

class DeformConvPack(DeformConv2d):
    def __init__(self, *args, **kwargs):
        super(DeformConvPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            2 * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return deform_conv2d(input = x, offset= offset, weight= self.weight, stride=_pair(self.stride),padding= _pair(self.padding), dilation=_pair(self.dilation))



if __name__ == '__main__':
    img = torch.randn(1, 32, 224, 224).cuda()
    model = DeformConvPack(32 ,64,3,padding =1).cuda()
    output = model(img)