#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_ssim import SSIMLoss
import torchvision
import numpy as np
#from fsim import FSIMLoss
from piq import FSIMLoss

    
class Fusionloss(nn.Module):
    def __init__(self,device):
        super(Fusionloss, self).__init__()
       
        self.FSIM=FSIMLoss(data_range=1.0,chromatic=False)
        
     
    def forward(self,image_vis,image_ir,generate_img,device):
        image_y=image_vis

        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        # y_grad,ir_grad=self.FSIM(image_y,image_ir)
       # y_grad=torch.abs(y_grad).to(device)
       # ir_grad=torch.abs(ir_grad).to(device)
        
       # print(torch.max(generate_img),torch.min(generate_img))
     #   generate_img_grad,_=self.FSIM(generate_img,generate_img)
        
       # x_grad_joint=torch.max(y_grad,ir_grad)
        
        #generate_img_grad=torch.abs( generate_img_grad).to(device)
        
     #   loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        #generate_img=torch.cat([generate_img,generate_img,generate_img],1)
        generate_img=torch.clamp(generate_img,min=0.00001,max=1.0)
       # x_in_max=torch.cat([x_in_max,x_in_max,x_in_max],1)
        print(torch.max(generate_img),torch.min(generate_img))
        print(torch.max(x_in_max),torch.min(x_in_max))
        loss_grad=self.FSIM(x_in_max,generate_img)
        
        loss_total=loss_in+10*loss_grad
       
        
        
        return loss_total,loss_in,loss_grad



if __name__ == '__main__':
    pass

