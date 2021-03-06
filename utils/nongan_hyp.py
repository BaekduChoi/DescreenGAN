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

class nongan :
    def __init__(self,json_dir,cuda=True,nch=32,num_blks_mid=12) :

        torch.autograd.set_detect_anomaly(True)

        self.params = read_json(json_dir)
        self.device = torch.device('cuda') if cuda else torch.device('cpu')

        self.netG = DescreenNetHyp(nch,num_blks_mid)

        self.netG = self.netG.to(self.device)

        self.perceptual_loss = PerceptualLoss(loss_type='l1',feature_layers=[2,3]).to(self.device)

    def getparams(self) :
        # reading the hyperparameter values from the json file
        self.lr = self.params['solver']['learning_rate']
        self.lr_step = self.params['solver']['lr_step']
        self.lr_gamma = self.params['solver']['lr_gamma']
        self.lambda_perceptual = self.params['solver']['lambda_perceptual']
        self.lambda_detail = self.params['solver']['lambda_detail']
        self.beta1 = self.params['solver']['beta1']
        self.beta2 = self.params['solver']['beta2']
        self.betas = (self.beta1,self.beta2)
        self.batch_size = self.params['datasets']['train']['batch_size']
    
    def getopts(self) :
        # set up the optimizers and schedulers
        self.optimizerG = optim.Adam(self.netG.parameters(),lr=self.lr,betas=self.betas,amsgrad=True)

        self.lr_sche_ftn = lambda epoch : 1.0 if epoch < self.lr_step else (self.lr_gamma)**(epoch-self.lr_step+1)
        self.schedulerG = optim.lr_scheduler.LambdaLR(self.optimizerG,self.lr_sche_ftn)

    def train(self) :
        trainloader, valloader = create_dataloaders(self.params)

        self.inittrain()
        # num_batches is saved for normalizing the running loss
        self.num_batches = len(trainloader)

        # starting iteration
        for epoch in range(self.start_epochs,self.epochs) :
            print('Epoch = '+str(epoch+1))
            print(self.schedulerG.get_last_lr())

            # training part of the iteration
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

                    loss_detail = self.fitG(GT,imgsH)
                    
                    # tqdm update
                    t.set_postfix_str('Losses - Detail : %.4f'%(loss_detail))
                    t.update()
                         
            print('Finished training for epoch '+str(epoch+1))
            
            if valloader is not None :
                print('Validating..')
                val_loss_detail = self.val(valloader,self.val_path,16,epoch)
                print('Validation result - detail loss = %.4f'%(val_loss_detail))

            self.schedulerG.step()

            self.train_losses.append([self.running_loss_detail])
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
        
        if self.pretrain and os.path.exists(self.pretrained_path+'losses.pkl') :
            losses_saved = pickle.load(open(self.pretrained_path+'losses.pkl','rb'))
            self.train_losses = losses_saved[0]
            self.val_losses = losses_saved[1]
        else :
            self.train_losses = []
            self.val_losses = []
    
    def val(self,testloader,test_dir,early_stop=None,epoch=None) :
        with torch.no_grad() :
            count = 0
            val_loss_detail = 0.
            for ii,data in enumerate(testloader) :
                inputH = data['halftone']
                inputH = inputH.to(self.device)

                GT = data['img'].to(self.device)

                # maximum img size for val is 1024x1024 due to memory issues
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
            torch.save({
                'epoch':epoch+1,
                'modelG_state_dict':self.netG.state_dict(),
                'optimizerG_state_dict':self.optimizerG.state_dict(),
                'schedulerG_state_dict':self.schedulerG.state_dict(),
            },path)

    
    def loadckp(self) :
        self.ckp_load = torch.load(self.params['solver']['ckp_path'])
        self.start_epochs = self.ckp_load['epoch']
        self.netG.load_state_dict(self.ckp_load['modelG_state_dict'])
        self.optimizerG.load_state_dict(self.ckp_load['optimizerG_state_dict'])
        self.schedulerG.load_state_dict(self.ckp_load['schedulerG_state_dict'])

        print('Resumed training - epoch '+str(self.start_epochs+1))

    def fitG(self,GT,imgsH) :

        detail = self.netG(imgsH)

        loss_detail = (self.lambda_perceptual*self.perceptual_loss(GT,detail)+self.lambda_detail*F.l1_loss(GT,detail))

        loss = loss_detail
        
        self.optimizerG.zero_grad()

        loss.backward()

        self.optimizerG.step()
        
        self.running_loss_detail += loss_detail.item()/self.num_batches

        return loss_detail.item()

    