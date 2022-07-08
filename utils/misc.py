#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Script for miscellaneous functions used
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

import json
import torch
from torch.utils.data import DataLoader, ConcatDataset
from data import HalftoneDataset, HalftoneDatasetTest
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
    Function that reads the components of the json file and returns a dataloader for test dataset
    Refer to test_naive.json for the structure of json file
    For test dataset we do not use data augmentation

    params : output of read_json(json_file_location)
"""
def create_test_dataloaders(params) :
    test_halftone_root = params["datasets"]["test"]["root_halftone"]
    test_img_type = params['datasets']['test']['img_type']
    n_workers = int(params['datasets']['test']['n_workers'])
    
    test_dataset = HalftoneDatasetTest(test_halftone_root,
                                    test_img_type)
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