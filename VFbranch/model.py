import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [Wavelet_ResBlock(out_channel, out_channel) for _ in range(num_res)]  # 8次残差，蓝色块

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [Wavelet_ResBlock(channel, channel) for _ in range(num_res)]  # 8次残差
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MFCF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MFCF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),  # 1x1卷积
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)  # 3x3卷积
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SFE(nn.Module):  # 浅卷积模块--从下采样图像中提取特征
    def __init__(self, out_plane):  # 浅卷积 # 调用就会执行这个，并且需要输出的通道数
        super(SFE, self).__init__()
        self.main = nn.Sequential(  # 顺序容器，按照加入的顺序构建单路径网络
            BasicConv(3, out_plane // 4, kernel_size=3, stride=1, relu=True),  # 普通的卷积+relu
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane - 3, kernel_size=1, stride=1, relu=True)  # 此1x1卷积的特征和输入Bk连接
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)  # 进一步细化连接之后的特征

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):  # 继承Module类  #特征注意模块--主动强调或抑制先前尺度的特征，并从SCM中学习特征的空间/通道重要性
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)  # 初始化属性,小型残差网络

    def forward(self, x1, x2):
        x = x1 * x2  # x1=EBout,x2=SCMout
        out = x1 + self.merge(x)  # EBout和卷积之后的特征进行融合
        return out


'''
网络中加入小波变换
'''


def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2#4,3,128,256
    x02 = x[:, :, 1::2, :] / 2#4,3,128,256
    x1 = x01[:, :, :, 0::2]#4,3,128,128
    x2 = x02[:, :, :, 0::2]#4,3,128,128
    x3 = x01[:, :, :, 1::2]#4,3,128,128
    x4 = x02[:, :, :, 1::2]#4,3,128,128
    x_LL = x1 + x2 + x3 + x4 #4,3,128,128
    x_HL = -x1 - x2 + x3 + x4#4,3,128,128
    x_LH = -x1 + x2 - x3 + x4#4,3,128,128
    x_HH = x1 - x2 - x3 + x4#4,3,128,128

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


# def dwt_init(x):
#     x01 = x[:, :, 0::2, :] / 2#4,3,128,256
#     x02 = x[:, :, 1::2, :] / 2#4,3,128,256
#     x1 = x01[:, :, :, 0::2]
#     x2 = x02[:, :, :, 0::2]
#     x3 = x01[:, :, :, 1::2]
#     x4 = x02[:, :, :, 1::2]
#     x_LL = x1 + x2 + x3 + x4
#     x_HL = -x1 - x2 + x3 + x4
#     x_LH = -x1 + x2 - x3 + x4
#     x_HH = x1 - x2 - x3 + x4
#
#     return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)
#
#
# def iwt_init(x):
#     r = 2
#     in_batch, in_channel, in_height, in_width = x.size()
#     # print([in_batch, in_channel, in_height, in_width])
#     out_batch, out_channel, out_height, out_width = in_batch, int(
#         in_channel / (r ** 2)), r * in_height, r * in_width
#     x1 = x[:, 0:out_channel, :, :] / 2
#     x2 = x[:, out_channel:out_channel * 2, :, :] / 2
#     x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
#     x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
#
#     h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
#
#     h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
#     h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
#     h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
#     h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
#
#     return h




# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)



class RSAM(nn.Module): 
    def __init__(self, inplanes, outplanes):
        super(RSAM, self).__init__() 
        midplanes = int(outplanes // 2) 

        self.pool_1_h = nn.AdaptiveAvgPool2d((None, 1))  
        self.pool_1_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_1_h = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False) 
        self.conv_1_w = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)  
        self.pool_3_h = nn.AdaptiveAvgPool2d((None, 3))
        self.pool_3_w = nn.AdaptiveAvgPool2d((3, None))
        self.conv_3_h = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False) 
        self.conv_3_w = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.pool_5_h = nn.AdaptiveAvgPool2d((None, 5))
        self.pool_5_w = nn.AdaptiveAvgPool2d((5, None))
        self.conv_5_h = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.conv_5_w = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.pool_7_h = nn.AdaptiveAvgPool2d((None, 7))
        self.pool_7_w = nn.AdaptiveAvgPool2d((7, None))
        self.conv_7_h = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.conv_7_w = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.fuse_conv = nn.Conv2d(midplanes * 4, midplanes, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)
     
        self.conv_final = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
       
        self.mask_conv_1 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)
        self.mask_relu = nn.ReLU(inplace=False)
        self.mask_conv_2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)

    def forward(self, x):
        _, _, h, w = x.size()
       

        x_1_h = self.pool_1_h(x)  
        x_1_h = self.conv_1_h(x_1_h)
        x_1_h = x_1_h.expand(-1, -1, h, w)  


        x_1_w = self.pool_1_w(x)  
        x_1_w = self.conv_1_w(x_1_w) 
        x_1_w = x_1_w.expand(-1, -1, h, w)
        # x2 = F.interpolate(x2, (h, w))

        x_3_h = self.pool_3_h(x)
        x_3_h = self.conv_3_h(x_3_h)
        x_3_h = F.interpolate(x_3_h, (h, w)) 

        x_3_w = self.pool_3_w(x)
        x_3_w = self.conv_3_w(x_3_w)
        x_3_w = F.interpolate(x_3_w, (h, w))

        x_5_h = self.pool_5_h(x)
        x_5_h = self.conv_5_h(x_5_h)
        x_5_h = F.interpolate(x_5_h, (h, w))

        x_5_w = self.pool_5_w(x)
        x_5_w = self.conv_5_w(x_5_w)
        x_5_w = F.interpolate(x_5_w, (h, w))

        x_7_h = self.pool_7_h(x)
        x_7_h = self.conv_7_h(x_7_h)
        x_7_h = F.interpolate(x_7_h, (h, w))

        x_7_w = self.pool_7_w(x)
        x_7_w = self.conv_7_w(x_7_w)
        x_7_w = F.interpolate(x_7_w, (h, w)) 

      
        hx = self.relu(self.fuse_conv(torch.cat((x_1_h + x_1_w, x_3_h + x_3_w, x_5_h + x_5_w, x_7_h + x_7_w), dim=1)))
        mask_1 = self.conv_final(hx).sigmoid() 
        out1 = x * mask_1 

        hx = self.mask_relu(self.mask_conv_1(out1))
        mask_2 = self.mask_conv_2(hx).sigmoid()
        hx = out1 * mask_2 

        return hx
    
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


class VFbranch(nn.Module):
    def __init__(self, base_channel):
        super(VFbranch, self).__init__()
        num_res=8
 
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),  
            EBlock(base_channel * 2, num_res), 
            EBlock(base_channel * 4, num_res), 
        ])
        """
        加BA模块
        """
        self.en_layer1 = nn.Sequential(  
            BasicConv(base_channel, base_channel * 4, kernel_size=3, relu=True, stride=2)

        )

        self.en_layer2 = nn.Sequential(
            BasicConv(base_channel * 4, base_channel * 8, kernel_size=3, relu=True, stride=2)

        )
        
        self.BA = RSAM(base_channel * 8, base_channel * 8) 
        self.de_layer3 = nn.Sequential(  
            BasicConv(base_channel * 8, base_channel * 4, kernel_size=4, relu=True, stride=2, transpose=True)
        )

        self.de_layer4 = nn.Sequential(  
            BasicConv(base_channel * 4, base_channel, kernel_size=4, relu=True, stride=2, transpose=True)
        )

        self.feat_extract = nn.ModuleList([ 
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1), 
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),  
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True), 
            BasicConv(base_channel, 1, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([ 
            DBlock(base_channel * 4, num_res), 
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([ 
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList( 
            [
                BasicConv(base_channel * 4, 1, kernel_size=3, relu=False, stride=1), 
                BasicConv(base_channel * 2, 1, kernel_size=3, relu=False, stride=1), 
            ]
        )

        self.AFFs = nn.ModuleList([ 
            MFCF(base_channel * 7, base_channel * 1), 
            MFCF(base_channel * 7, base_channel * 2) 
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SFE(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SFE(base_channel * 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.hu1 = hu(base_channel, base_channel)
        self.hu2 = hu(base_channel * 2, base_channel * 2)
        self.hu3 = hu(base_channel * 4, base_channel * 4)


    def forward(self, fa11, fa21, fa31): 
        x_ = fa11
        z2 = fa21 
        z4 = fa31  

        outputs = list()
        res1 = self.Encoder[0](x_) 

        z = self.feat_extract[1](res1) 
        z = self.FAM2(z, z2) 
        res2 = self.Encoder[1](z) 

        z = self.feat_extract[2](res2) 
        z = self.FAM1(z, z4) 
        z = self.Encoder[2](z)  
        # a = self.IWT(z)

        z12 = F.interpolate(res1, scale_factor=0.5) 
        z21 = F.interpolate(res2, scale_factor=2) 
        z42 = F.interpolate(z, scale_factor=2) 
        z41 = F.interpolate(z42, scale_factor=2) 

        res2 = self.AFFs[1](z12, res2, z42) 
        res1 = self.AFFs[0](res1, z21, z41) 

        res1 = self.en_layer1(res1) 
        in_feature = self.en_layer2(res1) 

        BA_out = self.BA(in_feature) 

        res1 = self.de_layer3(BA_out) 
        res1 = self.de_layer4(res1) 

        z = self.Decoder[0](z)  

        z_ = self.ConvsOut[0](z)  
        z = self.feat_extract[3](z) 
        
        outputs.append(z_) 

        z = torch.cat([z, res2], dim=1)
        # z = self.DWT(z)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)

        
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)

       
        outputs.append(z_)

        z = torch.cat([z, res1], dim=1)
       
        z = self.Convs[1](z)
        z = self.Decoder[2](z)

       
        z = self.feat_extract[5](z) 
       
        outputs.append(z)

        return outputs

