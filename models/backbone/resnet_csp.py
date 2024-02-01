import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

from ..utils import Conv_BN_ReLU

__all__ = ['resnet18', 'resnet50', 'resnet101']

base_url = 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/'
model_urls = {
    'resnet18': base_url + 'resnet18-imagenet.pth',
    'resnet50': base_url + 'resnet50-imagenet.pth',
    'resnet101': base_url + 'resnet101-imagenet.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Convkxk(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 padding=0):
        super(Convkxk, self).__init__()
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


class Res_layer(nn.Module):
    def __init__(self, block, planes, blocks, inplanes, stride=1):
        super(Res_layer, self).__init__()

        self.inplanes = inplanes

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        self.layers = []
        self.layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            self.layers.append(block(self.inplanes, planes))

        self.layers = nn.ModuleList(self.layers)


    def forward(self, x):

        f_s = []
        for layer in self.layers:
            x = layer(x)    
            f_s.append(x)

        result = f_s[0]
        for i in range(1, len(f_s)):
            result += f_s[i]

        return result

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        print('resnet_csp')
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.c1 = Conv_BN_ReLU(3, 32)
        self.c2 = Conv_BN_ReLU(3, 64)
        self.c3 = Conv_BN_ReLU(3, 128)
        self.c4 = Conv_BN_ReLU(3, 256)


        # self.layer1 = Res_layer(block, 64, layers[0], inplanes=128)
        # self.layer2 = Res_layer(block, 128, layers[1], stride=2, inplanes=64)
        # self.layer3 = Res_layer(block, 256, layers[2], stride=2, inplanes=128)
        # self.layer4 = Res_layer(block, 512, layers[3], stride=2, inplanes=256)
        
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        ori = x

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        f = []
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        c1 = self.c1(F.adaptive_avg_pool2d(ori, (l1.shape[2], l1.shape[3])))
        c2 = self.c2(F.adaptive_avg_pool2d(ori, (l2.shape[2], l2.shape[3])))
        c3 = self.c3(F.adaptive_avg_pool2d(ori, (l3.shape[2], l3.shape[3])))
        c4 = self.c4(F.adaptive_avg_pool2d(ori, (l4.shape[2], l4.shape[3])))

        f.append(torch.cat((l1, c1), 1))
        f.append(torch.cat((l2, c2), 1))
        f.append(torch.cat((l3, c3), 1))
        f.append(torch.cat((l4, c4), 1))

        return tuple(f)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        # return x

def reduce_channel(op):

    if type(op).__name__ == 'Conv2d':

        # ori_weight = op.weight.clone()

        # w_s = ori_weight.view(-1, op.kernel_size[0] * op.kernel_size[1])
        # w_s = sorted(w_s, key=sum, reverse=True)
        # w_s = torch.FloatTensor([item.detach().numpy() for item in w_s])
        # w_s = w_s.view(op.out_channels, op.in_channels, op.kernel_size[0], op.kernel_size[1])

        if op.in_channels != 3:
            op.in_channels //= 2
        op.out_channels //= 2

        op.weight.data = op.weight.data[:op.out_channels, :op.in_channels]
        # op.weight.data = w_s[:op.out_channels, :op.in_channels]

    if type(op).__name__ == 'BatchNorm2d':

        # ori_weight = op.weight.clone()
        # ori_bias = op.bias.clone()

        # w_s, w_index = torch.sort(ori_weight, dim=0, descending=True)
        # b_s = ori_bias[w_index]

        op.num_features //= 2

        op.weight.data = op.weight.data[:op.num_features]
        op.bias.data = op.bias.data[:op.num_features]
        op.running_mean.data = op.running_mean.data[:op.num_features]
        op.running_var.data = op.running_var.data[:op.num_features]
        # op.weight.data = w_s[:op.num_features]
        # op.bias.data = b_s[:op.num_features]
        # op.running_mean.data = w_s[:op.num_features]
        # op.running_var.data = b_s[:op.num_features]

    return op

def resnet18_to_csp(model):
    start_time = time.time()
    print("Creating CSP model...")

    # stem
    model.conv1 = reduce_channel(model.conv1)
    model.bn1 = reduce_channel(model.bn1)
    model.conv2 = reduce_channel(model.conv2)
    model.bn2 = reduce_channel(model.bn2)
    model.conv3 = reduce_channel(model.conv3)
    model.bn3 = reduce_channel(model.bn3)

    # layer 1
    model.layer1[0].conv1 = reduce_channel(model.layer1[0].conv1)
    model.layer1[0].bn1 = reduce_channel(model.layer1[0].bn1)
    model.layer1[0].conv2 = reduce_channel(model.layer1[0].conv2)
    model.layer1[0].bn2 = reduce_channel(model.layer1[0].bn2)
    model.layer1[0].downsample[0] = reduce_channel(model.layer1[0].downsample[0])
    model.layer1[0].downsample[1] = reduce_channel(model.layer1[0].downsample[1])

    model.layer1[1].conv1 = reduce_channel(model.layer1[1].conv1)
    model.layer1[1].bn1 = reduce_channel(model.layer1[1].bn1)
    model.layer1[1].conv2 = reduce_channel(model.layer1[1].conv2)
    model.layer1[1].bn2 = reduce_channel(model.layer1[1].bn2)
    # layer 2
    model.layer2[0].conv1 = reduce_channel(model.layer2[0].conv1)
    model.layer2[0].bn1 = reduce_channel(model.layer2[0].bn1)
    model.layer2[0].conv2 = reduce_channel(model.layer2[0].conv2)
    model.layer2[0].bn2 = reduce_channel(model.layer2[0].bn2)
    model.layer2[0].downsample[0] = reduce_channel(model.layer2[0].downsample[0])
    model.layer2[0].downsample[1] = reduce_channel(model.layer2[0].downsample[1])

    model.layer2[1].conv1 = reduce_channel(model.layer2[1].conv1)
    model.layer2[1].bn1 = reduce_channel(model.layer2[1].bn1)
    model.layer2[1].conv2 = reduce_channel(model.layer2[1].conv2)
    model.layer2[1].bn2 = reduce_channel(model.layer2[1].bn2)
    # layer 3
    model.layer3[0].conv1 = reduce_channel(model.layer3[0].conv1)
    model.layer3[0].bn1 = reduce_channel(model.layer3[0].bn1)
    model.layer3[0].conv2 = reduce_channel(model.layer3[0].conv2)
    model.layer3[0].bn2 = reduce_channel(model.layer3[0].bn2)
    model.layer3[0].downsample[0] = reduce_channel(model.layer3[0].downsample[0])
    model.layer3[0].downsample[1] = reduce_channel(model.layer3[0].downsample[1])

    model.layer3[1].conv1 = reduce_channel(model.layer3[1].conv1)
    model.layer3[1].bn1 = reduce_channel(model.layer3[1].bn1)
    model.layer3[1].conv2 = reduce_channel(model.layer3[1].conv2)
    model.layer3[1].bn2 = reduce_channel(model.layer3[1].bn2)
    # layer 4
    model.layer4[0].conv1 = reduce_channel(model.layer4[0].conv1)
    model.layer4[0].bn1 = reduce_channel(model.layer4[0].bn1)
    model.layer4[0].conv2 = reduce_channel(model.layer4[0].conv2)
    model.layer4[0].bn2 = reduce_channel(model.layer4[0].bn2)
    model.layer4[0].downsample[0] = reduce_channel(model.layer4[0].downsample[0])
    model.layer4[0].downsample[1] = reduce_channel(model.layer4[0].downsample[1])

    model.layer4[1].conv1 = reduce_channel(model.layer4[1].conv1)
    model.layer4[1].bn1 = reduce_channel(model.layer4[1].bn1)
    model.layer4[1].conv2 = reduce_channel(model.layer4[1].conv2)
    model.layer4[1].bn2 = reduce_channel(model.layer4[1].bn2)

    print('Finished. Cost time: ' + str(time.time() - start_time))

    return model


def resnet18_csp(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet18']), strict=False)
    model = resnet18_to_csp(model)
    return model


# def resnet50(pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on Places
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
#     return model


# def resnet101(pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on Places
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(load_url(model_urls['resnet101']), strict=False)
#     return model


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)
