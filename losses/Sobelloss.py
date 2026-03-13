#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import kornia.filters as KF



class Fusionloss(nn.Module):
    def __init__(self,device):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy(device) 
        self.device=device
        
   
    def forward(self,image_vis,image_ir,generate_img):

        x_in_max=torch.max(image_vis,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        
        vis_grad=self.sobelconv(image_vis).to(self.device)
        ir_grad=self.sobelconv(image_ir).to(self.device)
        generate_img_grad=self.sobelconv(generate_img)
        
        x_grad_joint=torch.max(vis_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
       
        loss_total=loss_in+10*loss_grad  
        
        
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self,device):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)
    
def Re_loss(img1, img2, weights=[5, 5], mask=None, ir_flag=False):
    ## img1 is the reconstructed image, img2 is the reference image
    grad1 =  KF.spatial_gradient(img1, order=2).abs().sum(dim=[1,2])
    grad2 = KF.spatial_gradient(img2, order=2).abs().sum(dim=[1,2])
    loss_intensity = F.l1_loss(img1, img2)
    loss_grad = F.l1_loss(grad1, grad2)
    loss_total = weights[0] * loss_intensity + weights[1] * loss_grad
    return loss_total, loss_intensity, loss_grad

if __name__ == '__main__':
    pass

