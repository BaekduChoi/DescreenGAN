#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:35:06 2020

@author: baekduchoi
"""

"""
    Script for miscellaneous functions used
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

import json
import torch
from torch.utils.data import DataLoader, ConcatDataset
from data import HalftoneDataset, screenImage, readScreen
from torch.nn import functional as F
from torch.nn import init as init
from torch import nn
import torchvision

import cv2
import scipy.signal
import numpy as np

"""
    Function that reads the json file and generates the dataloader to be used
    Only generates training and validation dataloader
"""
def create_dataloaders(params) :
    train_img_root = params["datasets"]["train"]["root_img"]
    train_halftone_root = params["datasets"]["train"]["root_halftone"]
    batch_size = int(params["datasets"]["train"]["batch_size"])
    train_img_type = params['datasets']['train']['img_type']
    n_workers = int(params['datasets']['train']['n_workers'])
    train_use_aug = params['datasets']['train']['use_aug']
    
    val_img_root = params["datasets"]["val"]["root_img"]
    val_halftone_root = params["datasets"]["val"]["root_halftone"]
    val_img_type = params['datasets']['val']['img_type']
    
    train_dataset = HalftoneDataset(train_img_root,
                                        train_halftone_root,
                                        train_img_type,
                                        train_use_aug)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=n_workers,
                                  shuffle=True)
    
    # no need to use augmentation for validation data
    val_dataset = HalftoneDataset(val_img_root,
                                        val_halftone_root,
                                        val_img_type)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                num_workers=n_workers,
                                shuffle=False)
    
    return train_dataloader, val_dataloader

"""
    Added extra smooth patch image / halftone pairs
"""
def create_dataloaders_extra(params) :
    train_img_root = params["datasets"]["train"]["root_img"]
    train_halftone_root = params["datasets"]["train"]["root_halftone"]
    batch_size = int(params["datasets"]["train"]["batch_size"])
    train_img_type = params['datasets']['train']['img_type']
    n_workers = int(params['datasets']['train']['n_workers'])
    train_use_aug = params['datasets']['train']['use_aug']
    
    val_img_root = params["datasets"]["val"]["root_img"]
    val_halftone_root = params["datasets"]["val"]["root_halftone"]
    val_img_type = params['datasets']['val']['img_type']
    
    train_dataset1 = HalftoneDataset(train_img_root,
                                        train_halftone_root,
                                        train_img_type,
                                        train_use_aug)
    train_dataset2 = HalftoneDataset('./img_patch/',
                                        './halftone_patch/',
                                        '.png',
                                        train_use_aug)
    train_dataset = ConcatDataset([train_dataset1,train_dataset2])
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=n_workers,
                                  shuffle=True)
    
    # no need to use augmentation for validation data
    val_dataset = HalftoneDataset(val_img_root,
                                        val_halftone_root,
                                        val_img_type)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                num_workers=n_workers,
                                shuffle=False)
    
    return train_dataloader, val_dataloader

"""
    Function that reads the components of the json file and returns a dataloader for test dataset
    Refer to test_naive.json for the structure of json file
    For test dataset we do not use data augmentation

    params : output of read_json(json_file_location)
"""
def create_test_dataloaders(params) :
    test_img_root = params["datasets"]["test"]["root_img"]
    test_halftone_root = params["datasets"]["test"]["root_halftone"]
    test_img_type = params['datasets']['test']['img_type']
    n_workers = int(params['datasets']['test']['n_workers'])
    
    test_dataset = HalftoneDataset(test_img_root,
                                    test_halftone_root,
                                    test_img_type,
                                    False)
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=1,
                                  num_workers=n_workers,
                                  shuffle=False)
    
    return test_dataloader

"""
    Function that reads the json file
"""
def read_json(json_dir) : 
    with open(json_dir,'r') as f :
        params = json.load(f)
    return params

"""
    Nasanen's HVS model
"""
class HVS(object) :
    
    def __init__(self) :
        N = 23
        c = 0.525
        d = 3.91
        G = 11.0
        pi = np.pi
        fs = pi*3500.0/180.0
        k = fs/(c*np.log(G)+d)
        
        self.hvs = np.zeros((2*N+1,2*N+1))
        
        for i in range(2*N+1) :
            for j in range(2*N+1) :
                m = i-N
                n = j-N
                
                denom = ((k**2)+4.0*(pi**2)*((m**2)+(n**2)))**1.5                
                val = 2.0*pi*k/denom
                
                dist = (float(m)**2.0+float(n)**2.0)**0.5
                if dist > float(N) :
                    self.hvs[i][j] = 0.0
                else :
                    self.hvs[i][j] = val*(float(N)+1-dist)
                
        # print(np.sum(self.hvs)**2)
        self.hvs = self.hvs/np.sum(self.hvs)
        self.N = N
    
    def __getitem__(self, keys) :
        m = keys[0]+self.N
        n = keys[1]+self.N
        return self.hvs[m][n]
    
    def getHVS(self) :
        return self.hvs.astype(np.float32)
    
    def size(self) :
        return self.hvs.shape

"""
    HVS error loss function
"""
def HVSloss(img1,img2,hvs) :
    k = hvs.size(2)
    M = img1.size(2)
    N = img1.size(3)

    pd = (k-1)//2

    img1p = F.pad(img1,(pd,pd,pd,pd),mode='circular')
    img2p = F.pad(img2,(pd,pd,pd,pd),mode='circular')
    img1_filtered = F.conv2d(img1p,hvs)
    img2_filtered = F.conv2d(img2p,hvs)

    return F.mse_loss(img1_filtered,img2_filtered)

"""
    Code adapted from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
"""
class PerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, loss_type = 'mse', feature_layers=[3]):
        super().__init__()

        assert loss_type in ['mse','l1']
        self.loss = F.l1_loss if loss_type == 'l1' else F.mse_loss

        # self.bl = (torchvision.models.vgg19(pretrained=True).features[:27].eval())
        # for p in self.bl.parameters():
        #     p.requires_grad = False

        blocks = []
        blocks.append(torchvision.models.vgg19(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[9:18].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[18:27].eval())

        for bl in blocks :
            for p in bl.parameters():
                p.requires_grad = False
        
        self.blocks = torch.nn.ModuleList(blocks)
        self.feature_layers = feature_layers

        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        x = input
        y = target

        loss = 0.

        for i,block in enumerate(self.blocks) :
            x = block(x)
            y = block(y)

            if i in self.feature_layers :
                loss += self.loss(x,y)

        return loss

"""
    Borrowed from https://github.com/xinntao/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py
"""
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.
    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)