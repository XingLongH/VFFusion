import os
import torch
import argparse
from torch.utils.data import DataLoader
from losses.Sobelloss import Fusionloss, Re_loss
from models.block.Drop import dropblock_step
from util.TaskFusion_dataset import Fusion_dataset
from util.common import check_dirs, init_seed, gpu_info, CosOneCycle
from fusion_model import ImageFusion
import torch.nn.functional as F
from losses.ssim import SSIM

def blur_image(input_im):
    blurred_im = F.avg_pool2d(input_im, kernel_size=3, stride=1, padding=1)
    return blurred_im

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out
def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def train(opt,device):
    init_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    gpu_info()  # 打印GPU信息
    save_path = check_dirs()

    train_dataset = Fusion_dataset('train')
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    model = ImageFusion(opt).cuda()
    criterion = Fusionloss(device)
    criterion1 = torch.nn.L1Loss()
    ssim_loss = SSIM()

    if opt.finetune:
        params = [{"params": [param for name, param in model.named_parameters()
                              if "backbone" in name], "lr": opt.learning_rate / 10},
                  {"params": [param for name, param in model.named_parameters()
                              if "backbone" not in name], "lr": opt.learning_rate}] 
        print("Using finetune for model")
    else:
        params = model.parameters()
    #optimizer = torch.optim.AdamW(params, lr=opt.learning_rate, weight_decay=0.001)
    optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
  
    #scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate, epochs=opt.epochs, up_rate=0)
   

    # best_loss=0
    for epoch in range(opt.epochs):
        model.train()
        for it, (image_vis, image_ir, name) in enumerate(train_loader):

            if epoch < 20:
                A = 0

                model.train()
                image_vis = image_vis.cuda()
            
                image_vis_ycrcb = RGB2YCrCb(image_vis).cuda()[:,0,:,:]
                image_vis_ycrcb = image_vis_ycrcb .unsqueeze(1)
                image_vis_ycrcb1 = blur_image(image_vis_ycrcb)
                image_ir = image_ir.cuda()
                image_ir1 = blur_image(image_ir)

                # image_vis_ycrcb1 = F.interpolate(image_vis_ycrcb, scale_factor=0.5, mode='bilinear') 
                # image_ir1 = F.interpolate(image_ir, scale_factor=0.5, mode='bilinear') 

                #label = label.cuda()
                label_img21 = F.interpolate(image_vis_ycrcb, scale_factor=0.5, mode='bilinear') 
                label_img41 = F.interpolate(image_vis_ycrcb, scale_factor=0.25, mode='bilinear') 

                label_img22 = F.interpolate(image_ir, scale_factor=0.5, mode='bilinear') 
                label_img42 = F.interpolate(image_ir, scale_factor=0.25, mode='bilinear') 

                # re_vi, re_ir, logits, outputs = model(image_vis_ycrcb.cuda(), image_ir)
                logits, outputs1, outputs2 = model(image_vis_ycrcb1.cuda(), image_ir1, A)
                
                optimizer.zero_grad()

                l11 = criterion1(outputs1[0], label_img41)  
                l21 = criterion1(outputs1[1], label_img21)
                l31 = criterion1(outputs1[2], image_vis_ycrcb)
                
                S11 = -ssim_loss(outputs1[0], label_img41)  
                S21 = -ssim_loss(outputs1[1], label_img21)
                S31 = -ssim_loss(outputs1[2], image_vis_ycrcb)

                loss_content1_vi = l11 + l21 + l31
                loss_content2_vi = S11 + S21 + S31
                loss_content_vi = loss_content1_vi + loss_content2_vi



                label_fft11 = torch.fft.fft2(label_img41, dim=(-2, -1))
                pred_fft11 = torch.fft.fft2(outputs1[0], dim=(-2, -1))
                label_fft21 = torch.fft.fft2(label_img21, dim=(-2, -1))
                pred_fft21 = torch.fft.fft2(outputs1[1], dim=(-2, -1))
                label_fft31 = torch.fft.fft2(image_vis_ycrcb, dim=(-2, -1))
                pred_fft31 = torch.fft.fft2(outputs1[2], dim=(-2, -1))

                f11 = criterion1(pred_fft11, label_fft11) 
                f21 = criterion1(pred_fft21, label_fft21)
                f31 = criterion1(pred_fft31, label_fft31)
                loss_fft1 = f11 + f21 + f31

                l12 = criterion1(outputs2[0], label_img42) 
                l22 = criterion1(outputs2[1], label_img22)
                l32 = criterion1(outputs2[2], image_ir)
                
                S12 = -ssim_loss(outputs2[0], label_img42) 
                S22 = -ssim_loss(outputs2[1], label_img22)
                S32 = -ssim_loss(outputs2[2], image_ir)
                loss_content1_ir = l12 + l22 + l32
                loss_content2_ir = S12 + S22 + S32
                loss_content_ir = loss_content1_ir + 10*loss_content2_ir

                label_fft12 = torch.fft.fft2(label_img42, dim=(-2, -1))
                pred_fft12 = torch.fft.fft2(outputs2[0], dim=(-2, -1))
                label_fft22 = torch.fft.fft2(label_img22, dim=(-2, -1))
                pred_fft22 = torch.fft.fft2(outputs2[1], dim=(-2, -1))
                label_fft32 = torch.fft.fft2(image_ir, dim=(-2, -1))
                pred_fft32 = torch.fft.fft2(outputs2[2], dim=(-2, -1))

                f12 = criterion1(pred_fft12, label_fft12)
                f22 = criterion1(pred_fft22, label_fft22)
                f32 = criterion1(pred_fft32, label_fft32)
                loss_fft2 = f12 + f22 + f32
    
                loss = loss_content_vi + loss_content_ir + 0.1 * loss_fft1 +  0.1 * loss_fft2

                
            
                #--------------- fusion loss
                loss_fusion, loss_in, loss_grad =criterion(image_vis_ycrcb.cuda(), image_ir, logits)
                loss_total = 0.5 * loss_fusion  + loss

            else:
                A = 1
                model.train()
                image_vis = image_vis.cuda()
            
                image_vis_ycrcb = RGB2YCrCb(image_vis).cuda()[:,0,:,:]
                image_vis_ycrcb = image_vis_ycrcb .unsqueeze(1)
                # image_vis_ycrcb1 = blur_image(image_vis_ycrcb)
                image_ir = image_ir.cuda()
                # image_ir1 = blur_image(image_ir)

                # image_vis_ycrcb1 = F.interpolate(image_vis_ycrcb, scale_factor=0.5, mode='bilinear') 
                # image_ir1 = F.interpolate(image_ir, scale_factor=0.5, mode='bilinear') 

                #label = label.cuda()
                label_img21 = F.interpolate(image_vis_ycrcb, scale_factor=0.5, mode='bilinear') 
                label_img41 = F.interpolate(image_vis_ycrcb, scale_factor=0.25, mode='bilinear')  

                label_img22 = F.interpolate(image_ir, scale_factor=0.5, mode='bilinear') 
                label_img42 = F.interpolate(image_ir, scale_factor=0.25, mode='bilinear')

                # re_vi, re_ir, logits, outputs = model(image_vis_ycrcb.cuda(), image_ir)
                logits, outputs1, outputs2 = model(image_vis_ycrcb.cuda(), image_ir, A)

                optimizer.zero_grad()

                l11 = criterion1(outputs1[0], label_img41)  
                l21 = criterion1(outputs1[1], label_img21)
                l31 = criterion1(outputs1[2], image_vis_ycrcb)
                loss_content1_vi = l11 + l21 + l31

                S11 = -ssim_loss(outputs1[0], label_img41) 
                S21 = -ssim_loss(outputs1[1], label_img21)
                S31 = -ssim_loss(outputs1[2], image_vis_ycrcb)
                loss_content2_vi = S11 + S21 + S31
                loss_content_vi = loss_content1_vi + loss_content2_vi

                label_fft11 = torch.fft.fft2(label_img41, dim=(-2, -1))
                pred_fft11 = torch.fft.fft2(outputs1[0], dim=(-2, -1))
                label_fft21 = torch.fft.fft2(label_img21, dim=(-2, -1))
                pred_fft21 = torch.fft.fft2(outputs1[1], dim=(-2, -1))
                label_fft31 = torch.fft.fft2(image_vis_ycrcb, dim=(-2, -1))
                pred_fft31 = torch.fft.fft2(outputs1[2], dim=(-2, -1))

                f11 = criterion1(pred_fft11, label_fft11)  
                f21 = criterion1(pred_fft21, label_fft21)
                f31 = criterion1(pred_fft31, label_fft31)
                loss_fft1 = f11 + f21 + f31

                l12 = criterion1(outputs2[0], label_img42) 
                l22 = criterion1(outputs2[1], label_img22)
                l32 = criterion1(outputs2[2], image_ir)

                S12 = -ssim_loss(outputs2[0], label_img42) 
                S22 = -ssim_loss(outputs2[1], label_img22)
                S32 = -ssim_loss(outputs2[2], image_ir)
                loss_content1_ir = l12 + l22 + l32
                loss_content2_ir = S12 + S22 + S32
                loss_content_ir = loss_content1_ir + 10*loss_content2_ir

                label_fft12 = torch.fft.fft2(label_img42, dim=(-2, -1))
                pred_fft12 = torch.fft.fft2(outputs2[0], dim=(-2, -1))
                label_fft22 = torch.fft.fft2(label_img22, dim=(-2, -1))
                pred_fft22 = torch.fft.fft2(outputs2[1], dim=(-2, -1))
                label_fft32 = torch.fft.fft2(image_ir, dim=(-2, -1))
                pred_fft32 = torch.fft.fft2(outputs2[2], dim=(-2, -1))

                f12 = criterion1(pred_fft12, label_fft12)  
                f22 = criterion1(pred_fft22, label_fft22)
                f32 = criterion1(pred_fft32, label_fft32)
                loss_fft2 = f12 + f22 + f32
    
                loss = loss_content_vi + loss_content_ir + 0.1 * loss_fft1 +  0.1 *loss_fft2

                
            
                #--------------- fusion loss
                loss_fusion, loss_in, loss_grad =criterion(image_vis_ycrcb.cuda(), image_ir, logits)
                
                loss_total = loss_fusion  + 0.5 *loss

            
        
            loss_total.backward()

            optimizer.step()
            
            print('==Epoch:[{}],[{}]/[{}],loss_total:{},loss_in: {},loss_grad: {}'
                  .format(epoch,it,len(train_loader),loss_total.item(),loss_in.item(),loss_grad.item()))
            # if it==1:
            #    best_loss=loss_fusion
            # if it // 200:
            #     best_loss
            #     if loss_fusion>best_loss:
                
            # scheduler.step()
        # dropblock_step(model)

           # Save model and log
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(model.state_dict(), save_path+'/checkpoint_epoch_'+str(epoch)+'.pt')
     

        print('An epoch finished.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Change Detection train')

    # parser.add_argument("--backbone", type=str, default="mtc_64")
    parser.add_argument("--backbone", type=str, default="mtc_64")

    parser.add_argument("--neck", type=str, default="fpn+drop")
    parser.add_argument("--head", type=str, default="fcn")

    parser.add_argument("--pretrain", type=str,
                       default="")  
    parser.add_argument("--cuda", type=str, default="2")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
  
    parser.add_argument("--finetune", type=bool, default=True)
  

    opt = parser.parse_args()
    print("\n" + "-" * 30 + "OPT" + "-" * 30)
    print(opt)    
    device = torch.device("cuda:0")
    train(opt,device)


