import os
import re
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models.block.Base import ChannelChecker
from models.head.FCN import FCNHead
from models.neck.FPN import FPNNeck
from collections import OrderedDict
from typing import Dict
from SDEM import SDEMPLUS
from bra_unet_system import BiFormer
from VFbranch.model import VFbranch
   
mtc= BiFormer()

class S2M(nn.Module):
    '''
    Scene Specific Mask
    '''
    def __init__(self, channels, r=4):
        super(S2M, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels), 
            nn.ReLU(inplace=True),            
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels), 
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_block = nn.BatchNorm2d(channels)

    def forward(self, x):
        # spatial attention
        local_w = self.local_att(x) ## local attention
        ## channel attention
        global_w = self.global_att(x)
        mask = self.sigmoid(local_w * global_w)
        masked_feature = mask * x
        output = self.conv_block(masked_feature)
        return output

class SFP(nn.Module):
    '''
    Scene Fidelity Path
    '''
    def __init__(self, channels, img=False):        
        super(SFP, self).__init__()
        self.mask = S2M(channels[0])
        self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
        nn.Tanh(),
        )

    def forward(self, x):
        x = self.mask(x)
        return (self.conv_block(x) + 1) / 2
    
class IFP(nn.Module):
    '''
    Scene Fidelity Path
    '''
    def __init__(self, channels):        
        super(IFP, self).__init__()
        self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
        nn.Tanh(),
        )

    def forward(self, x):
        return (self.conv_block(x) + 1) / 2
    
class hu(nn.Module):
    '''
    Scene Fidelity Path
    '''
    def __init__(self, channels1, channels2):        
        super(hu, self).__init__()
        self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=channels1, out_channels=channels2, kernel_size=3, padding=1),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        return self.conv_block(x)
    
class long(nn.Module):
    '''
    Scene Fidelity Path
    '''
    def __init__(self, channels1, channels2):        
        super(long, self).__init__()
        self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=channels1, out_channels=channels2, kernel_size=3, padding=1),
        nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        return self.conv_block(x)

class long1(nn.Module):
    '''
    Scene Fidelity Path
    '''
    def __init__(self, channels1, channels2):        
        super(long1, self).__init__()
        self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=channels1, out_channels=channels2, kernel_size=3, padding=1),
        nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        return self.conv_block(x)

class ImageFusion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.inplanes = int(re.sub(r"\D", "", opt.backbone.split("_")[-1]))  # backbone的名称中必须在"_"之后加上它的通道数
  
        self._create_backbone(opt.backbone)
        self._create_neck(opt.neck)
        self._create_heads(opt.head)

        if opt.pretrain.endswith(".pt"):
            self._init_weight(opt.pretrain)   # todo:这里预训练初始化和 hrnet主干网络的初始化有冲突，必须要改！

        self.pred_vi = SFP([self.inplanes, 1])                
        self.pred_ir = SFP([self.inplanes, 1])
        self.pred_fusion = IFP([self.inplanes, 1])
        self.FCIM = FCIM(self.inplanes)
        self.VFbranch = VFbranch(self.inplanes)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.hu = hu(self.inplanes, self.inplanes*2)
        self.xing = nn.Conv2d(self.inplanes, self.inplanes*2, kernel_size=3, padding=1)
        self.long = long(self.inplanes, self.inplanes*2)
        self.long1 = long1(self.inplanes, self.inplanes*4)

 

    def forward(self, xa, xb, A):
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."
        
        fa1, fa2, fa3, fa4 = self.backboneA(xa)  
        fa11, fa21, fa31, fa41 = self.FCIM(fa1, fa2, fa3, fa4)
        outputs1 = self.VFbranch(fa11, fa21, fa31)

        fb1, fb2, fb3, fb4 = self.backboneB(xb)
        fb11, fb21, fb31, fb41 = self.FCIM(fb1, fb2, fb3, fb4)
        outputs2 = self.VFbranch(fb11, fb21, fb31)


        ms_feats = fa11, fa21, fa31, fa41, fb11, fb21, fb31, fb41   # 多尺度特征

        fusion = self.neck(ms_feats)

        # vi_img =  self.pred_vi(fusion)
        # ir_img =  self.pred_ir(fusion)
        # out = self.pred_fusion(fusion)
        out = self.head_forward(ms_feats, fusion, out_size=(h_input, w_input))

        if A==0:
            out1 = out

        else:
            x1 = fusion
            x2 = self.long(fusion)
            x3 = self.long1(fusion)
            out1 = self.MRDNet(x1, x2, x3)
            out1 = out1[2]


        return out1, outputs1, outputs2
        # return vi_img, ir_img, out, outputs

   


    def head_forward(self, ms_feats, fusion, out_size):
     
        # print(fusion.size())
        out = F.interpolate(self.head(fusion), size=out_size, mode='bilinear', align_corners=True)
      

        return out
    


    def _create_backbone(self, backbone):
        # if 'coat' in backbone:
        #     self.backboneA = coatnet_0()
        #     self.backboneB = coatnet_0()
            
        if 'mtc' in backbone:
            self.backboneA = mtc
            self.backboneB = mtc
        else:
            raise Exception('Not Implemented yet: {}'.format(backbone))

    def _create_neck(self, neck):
        if 'fpn' in neck:
            self.neck = FPNNeck(self.inplanes, neck)

    def _select_head(self, head):
        if head == 'fcn':
            return FCNHead(self.inplanes, 1)

    def _create_heads(self, head):
        self.head = self._select_head(head)
  


