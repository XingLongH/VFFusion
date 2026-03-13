import torch
import torch.nn as nn
import torch.nn.functional as F

# import numpy as np
# import os
# import math

# from vgg import VGG


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SDEM(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)] * 4)
            # nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1))

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i in range(len(xs)):
            if xs[i].shape[-1] > target_size:
                xs[i] = F.adaptive_avg_pool2d(xs[i], (target_size, target_size))
            elif xs[i].shape[-1] < target_size:
                xs[i] = F.interpolate(xs[i], size=(target_size, target_size),
                                  mode='bilinear', align_corners=True)
            
        for i in range(len(xs)):
            ans = ans * self.convs[3-i](xs[3-i])

        return ans


class FCIM(nn.Module):
    def __init__(self, inchannels):
        super(FCIM, self).__init__()
        #   SDEM
        self.ca_1 = ChannelAttention(inchannels)
        self.sa_1 = SpatialAttention()
        
        self.ca_2 = ChannelAttention(inchannels*2)
        self.sa_2 = SpatialAttention()
        
        self.ca_3 = ChannelAttention(inchannels*4)
        self.sa_3 = SpatialAttention()
        
        self.ca_4 = ChannelAttention(inchannels*8)
        self.sa_4 = SpatialAttention()
        
        self.Translayer_1 = BasicConv2d(inchannels,  inchannels, 1)
        self.Translayer_2 = BasicConv2d(inchannels*2, inchannels, 1)
        self.Translayer_3 = BasicConv2d(inchannels*4, inchannels, 1)
        self.Translayer_4 = BasicConv2d(inchannels*8, inchannels, 1)
        
        self.sdem1= SDEM(inchannels)
        self.sdem2= SDEM(inchannels)
        self.sdem3= SDEM(inchannels)
        self.sdem4= SDEM(inchannels)

    def forward(self, fa1,fa2,fa3,fa4):
        
        FA2 = fa2
        FA3 = fa3
        FA4 = fa4
        
        fa1 = self.ca_1(fa1) * fa1
        fa1 = self.sa_1(fa1) * fa1
        fa1 = self.Translayer_1(fa1)

        fa2 = self.ca_2(fa2) * fa2
        fa2 = self.sa_2(fa2) * fa2
        fa2 = self.Translayer_2(fa2)

        fa3 = self.ca_3(fa3) * fa3
        fa3 = self.sa_3(fa3) * fa3
        fa3 = self.Translayer_3(fa3)

        fa4 = self.ca_4(fa4) * fa4
        fa4 = self.sa_4(fa4) * fa4
        fa4 = self.Translayer_4(fa4)

        fa44 = self.sdem4([fa1, fa2, fa3, fa4], fa1)

        return fa44,FA2,FA3,FA4


