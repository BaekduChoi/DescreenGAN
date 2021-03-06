import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

from torch import nn
import torch
from torch import optim
from torch.nn import functional as F
from abc import ABC
from tqdm import tqdm
import numpy as np
import cv2
import argparse
import pickle

from network import *
from misc import *
from losses import *

class cRaGAN :
    def __init__(self,json_dir,cuda=True,nch=32,num_blks_mid=12) :

        torch.autograd.set_detect_anomaly(True)

        self.params = read_json(json_dir)
        self.device = torch.device('cuda') if cuda else torch.device('cpu')

        self.netG = DescreenNetHyp(nch,num_blks_mid)
        self.netD = Unet(in_ch=1, sn=True)

        self.netG = self.netG.to(self.device)
        self.netD = self.netD.to(self.device)

        self.perceptual_loss = PerceptualLoss(loss_type='l1',feature_layers=[2,3],before_act=True).to(self.device)

    def getparams(self) :
        # reading the hyperparameter values from the json file
        self.lr = self.params['solver']['learning_rate']
        self.lr_ratio = self.params['solver']['lr_ratio']
        self.lr_step = self.params['solver']['lr_step']
        self.lr_gamma = self.params['solver']['lr_gamma']
        self.lambda_perceptual = self.params['solver']['lambda_perceptual']
        self.lambda_detail = self.params['solver']['lambda_detail']
        self.lambda_adv = self.params['solver']['lambda_adv']
        self.beta1 = self.params['solver']['beta1']
        self.beta2 = self.params['solver']['beta2']
        self.betas = (self.beta1,self.beta2)
        self.batch_size = self.params['datasets']['train']['batch_size']
    
    def getopts(self) :
        self.optimizerD = optim.Adam(self.netD.parameters(),lr=self.lr*self.lr_ratio,betas=self.betas,amsgrad=True)
        self.optimizerG = optim.Adam(self.netG.parameters(),lr=self.lr,betas=self.betas,amsgrad=True)

        self.lr_sche_ftn = lambda epoch : 1.0 if epoch < self.lr_step else (self.lr_gamma)**(epoch-self.lr_step+1)
        self.schedulerD = optim.lr_scheduler.LambdaLR(self.optimizerD,self.lr_sche_ftn)
        self.schedulerG = optim.lr_scheduler.LambdaLR(self.optimizerG,self.lr_sche_ftn)

    def train(self) :
        trainloader, valloader = create_dataloaders(self.params)
        self.gan_loss = RaHingeGANLoss()

        self.inittrain()
        # num_batches is saved for normalizing the running loss
        self.num_batches = len(trainloader)

        # starting iteration
        for epoch in range(self.start_epochs,self.epochs) :
            print('Epoch = '+str(epoch+1))
            print(self.schedulerG.get_last_lr())

            # training part of the iteration
            self.running_loss_adv_G = 0.0
            self.running_loss_adv_D = 0.0
            self.running_loss_detail = 0.0

            # tqdm setup is borrowed from SRFBN github
            # https://github.com/Paper99/SRFBN_CVPR19
            with tqdm(total=len(trainloader),\
                    desc='Epoch: [%d/%d]'%(epoch+1,self.epochs),miniters=1) as t:
                for i,data in enumerate(trainloader) :
                    
                    GT = data['img']
                    imgsH = data['halftone']
                    GT = GT.to(self.device)
                    imgsH = imgsH.to(self.device)

                    loss_detail, loss_adv_G, gen_imgs = self.fitG(GT,imgsH)

                    # use a pool of generated images for training discriminator
                    if self.use_pool :
                        if self.batch_size == 1 :
                            output_pool = self.pooling_onesample(gen_imgs.detach())
                        else :
                            output_pool = self.pooling(gen_imgs.detach())
                    else :
                        output_pool = gen_imgs.detach()

                    loss_adv_D = self.fitD(GT,output_pool)
                    
                    # tqdm update
                    t.set_postfix_str('Losses - Detail : %.4f'%(loss_detail)+\
                                    ', adv_G : %.4f'%(loss_adv_G)+\
                                    ', adv_D : %.4f'%(loss_adv_D))
                    t.update()
                       
            print('Finished training for epoch '+str(epoch+1))
            
            if valloader is not None :
                print('Validating..')
                val_loss_detail = self.val(valloader,self.val_path,16,epoch)
                print('Validation result - detail loss = %.4f'%(val_loss_detail))

            self.schedulerG.step()
            self.schedulerD.step()

            self.train_losses.append([self.running_loss_detail,self.running_loss_adv_G,self.running_loss_adv_D])
            self.val_losses.append([val_loss_detail])

            self.saveckp(epoch)

        
    def inittrain(self) :
        self.getparams()
        self.getopts()

        # reading more hyperparameters and checkpoint saving setup
        self.epochs = self.params['solver']['num_epochs']
        self.save_ckp_step = self.params['solver']['save_ckp_step']
        self.pretrained_path = self.params['solver']['pretrained_path']
        self.val_path = self.params['solver']['val_path']
        self.use_pool = self.params['solver']['use_pool']

        if self.val_path[-1] != '/':
            self.val_path += '/'

        if not os.path.isdir(self.pretrained_path) :
            os.mkdir(self.pretrained_path)
        
        if not os.path.isdir(self.val_path) :
            os.mkdir(self.val_path)

        self.start_epochs = 0
        self.pretrain = self.params['solver']['pretrain']

        self.pool = None
        self.pool_size = 64*self.batch_size

        if self.pretrain :
            self.loadckp()    
            print(self.schedulerG.get_last_lr())
        else :
            self.load_prl_ckp()
        
        if self.pretrain and os.path.exists(self.pretrained_path+'losses.pkl') :
            losses_saved = pickle.load(open(self.pretrained_path+'losses.pkl','rb'))
            self.train_losses = losses_saved[0]
            self.val_losses = losses_saved[1]
        else :
            self.train_losses = []
            self.val_losses = []
    
    def test_final(self) :
        self.loadckp_test()

        testloader = create_test_dataloaders(self.params)
        test_path = self.params["solver"]["testpath"]
        if test_path[-1] != '/' :
            test_path += '/'

        if not os.path.isdir(test_path) :
            os.mkdir(test_path)

        self.test(testloader,test_path)
    
    def loadckp_test(self) :
        self.ckp_load = torch.load(self.params['solver']['ckp_path'])
        self.netG.load_state_dict(self.ckp_load['modelG_state_dict'])

    def test(self,testloader,test_dir) :
        with torch.no_grad() :
            count = 0
            with tqdm(total=len(testloader),\
                    desc='Testing.. ',miniters=1) as t:
                for ii,data in enumerate(testloader) :
                    inputH = data['halftone']
                    inputH = inputH.to(self.device)

                    H,W = inputH.shape[2], inputH.shape[3]

                    sy = (H%16)//2
                    sx = (W%16)//2
                    Ny = (H//16)*16
                    Nx = (W//16)*16

                    inputH = inputH[:,:,sy:sy+Ny,sx:sx+Nx]

                    outputs = self.netG(inputH)
                    
                    img_size1,img_size2 = outputs.shape[2], outputs.shape[3]
                    
                    for j in range(outputs.shape[0]) :
                        imgR = torch.zeros([img_size1,img_size2],dtype=torch.float32)
                        imgR[:,:] = outputs[j,:,:,:].squeeze()
                        imgR = imgR.detach().cpu().numpy()
                        imgR = np.clip(imgR,0,1)
                        imgBGR = (255*imgR).astype('uint8')
                        imname = test_dir+str(count+1)+'.png'
                        cv2.imwrite(imname,imgBGR)

                        imgR3 = torch.zeros([img_size1,img_size2],dtype=torch.float32)
                        imgR3[:,:] = inputH[j,:,:,:].squeeze()
                        imgR3 = imgR3.detach().cpu().numpy()
                        imgBGR3 = (255*imgR3).astype('uint8')
                        imname3 = test_dir+str(count+1)+'_dbs.png'
                        cv2.imwrite(imname3,imgBGR3)
                        
                        count += 1
                    # tqdm update
                    t.update()
    
    def val(self,testloader,test_dir,early_stop=None,epoch=None) :
        with torch.no_grad() :
            count = 0
            val_loss_detail = 0.
            for ii,data in enumerate(testloader) :
                inputH = data['halftone']
                inputH = inputH.to(self.device)

                GT = data['img'].to(self.device)

                _,_,H,W = inputH.shape
                if H > 1024 :
                    sy = (H-1024)//2
                    inputH = inputH[:,:,sy:sy+1024,:]
                    GT = GT[:,:,sy:sy+1024,:]
                if W > 1024 :
                    sx = (W-1024)//2
                    inputH = inputH[:,:,:,sx:sx+1024]
                    GT = GT[:,:,:,sx:sx+1024]

                outputs = self.netG(inputH)
                val_loss_detail += (self.lambda_perceptual*self.perceptual_loss(GT,outputs)+F.l1_loss(GT,outputs))/len(testloader)
                
                img_size1,img_size2 = outputs.shape[2], outputs.shape[3]

                if early_stop != None :
                    if count < early_stop :
                        for j in range(outputs.shape[0]) :
                            imgR = torch.zeros([img_size1,img_size2],dtype=torch.float32)
                            imgR[:,:] = outputs[j,:,:].squeeze()
                            imgR = imgR.detach().cpu().numpy()
                            imgR = np.clip(imgR,0,1)
                            imgBGR = (255*imgR).astype('uint8')
                            imname = test_dir+str(count+1)+'.png' if epoch == None else test_dir+'%d_e%d.png'%(count+1,epoch)
                            cv2.imwrite(imname,imgBGR)
                            
                            imgR2 = torch.zeros([img_size1,img_size2],dtype=torch.float32)
                            imgR2[:,:] = GT[j,:,:].squeeze()
                            imgR2 = imgR2.detach().cpu().numpy()
                            imgBGR2 = (255*imgR2).astype('uint8')
                            imname2 = test_dir+str(count+1)+'_GT.png'
                            cv2.imwrite(imname2,imgBGR2)
                            
                            count += 1
        return val_loss_detail
    
    def saveckp(self,epoch) :
        if (epoch+1)%self.save_ckp_step == 0 :
            path = self.pretrained_path+'/epoch'+str(epoch+1)+'.ckp'
            pickle.dump([self.train_losses,self.val_losses],open(self.pretrained_path+'losses.pkl','wb'))
            if self.use_pool :
                torch.save({
                    'epoch':epoch+1,
                    'modelG_state_dict':self.netG.state_dict(),
                    'optimizerG_state_dict':self.optimizerG.state_dict(),
                    'schedulerG_state_dict':self.schedulerG.state_dict(),
                    'modelD_state_dict':self.netD.state_dict(),
                    'optimizerD_state_dict':self.optimizerD.state_dict(),
                    'schedulerD_state_dict':self.schedulerD.state_dict(),
                    'pool':self.pool
                },path)
            else :
                torch.save({
                    'epoch':epoch+1,
                    'modelG_state_dict':self.netG.state_dict(),
                    'optimizerG_state_dict':self.optimizerG.state_dict(),
                    'schedulerG_state_dict':self.schedulerG.state_dict(),
                    'modelD1_state_dict':self.netD.state_dict(),
                    'optimizerD1_state_dict':self.optimizerD.state_dict(),
                    'schedulerD1_state_dict':self.schedulerD.state_dict(),
                },path)

    
    def loadckp(self) :
        self.ckp_load = torch.load(self.params['solver']['ckp_path'])
        self.start_epochs = self.ckp_load['epoch']
        self.netG.load_state_dict(self.ckp_load['modelG_state_dict'])
        self.netD.load_state_dict(self.ckp_load['modelD_state_dict'])
        self.optimizerG.load_state_dict(self.ckp_load['optimizerG_state_dict'])
        self.optimizerD.load_state_dict(self.ckp_load['optimizerD_state_dict'])
        self.schedulerG.load_state_dict(self.ckp_load['schedulerG_state_dict'])
        self.schedulerD.load_state_dict(self.ckp_load['schedulerD_state_dict'])

        if self.use_pool :
            self.pool = self.ckp_load['pool']

        print('Resumed training - epoch '+str(self.start_epochs+1))
    
    def load_prl_ckp(self) :
        self.ckp_load = torch.load(self.params['solver']['prl_ckp_path'])
        self.netG.load_state_dict(self.ckp_load['modelG_state_dict'])

    def fitG(self,GT,imgsH) :

        for _p in self.netD.parameters() :
            _p.requires_grad_(False)

        detail = self.netG(imgsH)
        loss_detail = self.lambda_perceptual*self.perceptual_loss(GT,detail)+self.lambda_detail*F.l1_loss(GT,detail)

        pred_fake = self.netD(detail)
        pred_real = self.netD(GT)
        loss_adv_G = self.gan_loss(pred_fake,pred_real)

        loss = loss_detail + self.lambda_adv*loss_adv_G
        
        self.optimizerG.zero_grad()

        loss.backward()

        self.optimizerG.step()

        self.running_loss_detail += loss_detail.item()/self.num_batches
        self.running_loss_adv_G += loss_adv_G.item()/self.num_batches

        return loss_detail.item(), loss_adv_G.item(), detail.detach()

    def pooling(self,output_vae_orig) :
        if self.pool is None :
            self.pool = output_vae_orig
            output_vae = output_vae_orig
        elif self.pool.shape[0] < self.pool_size :
            self.pool = torch.cat([self.pool,output_vae_orig],dim=0)
            output_vae = output_vae_orig
        else :
            temp = output_vae_orig
            batch_size = temp.shape[0]
            ridx = torch.randperm(batch_size)
            ridx2 = torch.randperm(self.pool.shape[0])
            output_vae = torch.cat((temp[ridx[:batch_size//2],:,:,:],self.pool[ridx2[:batch_size//2],:,:,:]),dim=0)
            self.pool = torch.cat((self.pool[ridx2[batch_size//2:],:,:,:],temp[ridx[batch_size//2:],:,:,:]),dim=0)
        return output_vae
    
    def pooling_onesample(self,output_vae_orig) :
        if self.pool is None :
            self.pool = output_vae_orig
            output_vae = output_vae_orig
        else :
            if self.pool.shape[0] < self.pool_size :
                self.pool = torch.cat([self.pool,output_vae_orig],dim=0)

            prob = np.random.random_sample()
            if prob >= 0.5 :
                rindex = np.random.randint(0,self.pool.size(0))
                output_vae = torch.unsqueeze(self.pool[rindex,:,:,:],0)
                self.pool[rindex,:,:,:] = output_vae_orig
            else :
                output_vae = output_vae_orig
        return output_vae
    
    def fitD(self,GT,output_pool) :
        
        for _p in self.netD.parameters() :
            _p.requires_grad_(True)

        pred_real = self.netD(GT)
        pred_fake = self.netD(output_pool)

        loss_adv_D = self.gan_loss(pred_real,pred_fake)

        self.optimizerD.zero_grad()

        loss_adv_D.backward()
        self.optimizerD.step()

        self.running_loss_adv_D += (loss_adv_D.item())/self.num_batches

        return loss_adv_D.item()

    