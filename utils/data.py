# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import os
import glob
import cv2
import numpy as np
import torch
from torch.nn import functional as F
# from matplotlib import pyplot as plt

"""
    data augmentation
    only using vertical and horizontal flip and 90 degree rotations
"""
def augmentImage(img,hflip,vflip,rot) :
    if hflip: img = img[:, ::-1]
    if vflip: img = img[::-1, :]
    if rot: img = img.transpose(1, 0)
    return img
    
"""
    Dataset class for halftoning
"""
class HalftoneDataset(Dataset) :
    def __init__(self,root_img,root_halftone,img_type='.png',augment=False) :
        self.root_img = root_img
        self.root_halftone = root_halftone
        self.img_type = img_type

        if self.root_img[-1] != '/' :
            self.root_img += '/'
        if self.root_halftone[-1] != '/' :
            self.root_halftone += '/'
            
        self.filenames = glob.glob(os.path.join(self.root_img,'*'+img_type))
        self.filenames_bases = []
        for _n in self.filenames :
            self.filenames_bases.append(os.path.splitext(os.path.basename(_n))[0])

        self.len = len(self.filenames)
        self.augment = augment
        # data augmentation makes the dataset x8 larger
        if self.augment :
            self.len = 8*self.len
        
    def __len__(self) :
        return self.len
    
    def __getitem__(self,idx) :

        # decide which permutation should be used for the given index
        # based on the index's residual wrt 8
        # generate both RGB and grayscale image
        if self.augment : 
            idx0 = idx//8
            r = idx-8*idx0
            rot = (r%2)==0
            hf = (r//4)>0
            vf = ((r//2)%2)==0
            img_name = self.root_img+self.filenames_bases[idx0]+self.img_type
            imgh_name = self.root_halftone+self.filenames_bases[idx0]+'h'+self.img_type
            imgG = cv2.imread(img_name,0)
            imgG = augmentImage(imgG,hf,vf,rot)
            imgG = imgG.astype(np.float32)/255.0
            imgH = cv2.imread(imgh_name,0)
            imgH = augmentImage(imgH,hf,vf,rot)
            imgH = imgH.astype(np.float32)/255.0
            
        # if no augmentation is used, directly use the image
        # generate both RGB and grayscale image
        else :
            img_name = self.root_img+self.filenames_bases[idx]+self.img_type
            imgh_name = self.root_halftone+self.filenames_bases[idx]+'h'+self.img_type
            imgG = cv2.imread(img_name,0)
            imgG = imgG.astype(np.float32)/255.0
            imgH = cv2.imread(imgh_name,0)
            imgH = imgH.astype(np.float32)/255.0
        
        # add a dimension for batch processing
        imgG = np.expand_dims(imgG,0)
        imgH = np.expand_dims(imgH,0)
        
        # convert to tensors
        imgG = torch.from_numpy(imgG)
        imgH = torch.from_numpy(imgH)
        
        # all the converted images are given as the data
        sample = {
            'img':imgG,
            'halftone':imgH
        }
        
        return sample

"""
    Dataset class for halftoning, testing only
"""
class HalftoneDatasetTest(Dataset) :
    def __init__(self,root_halftone,img_type='.png') :
        self.root_halftone = root_halftone
        self.img_type = img_type

        if self.root_halftone[-1] != '/' :
            self.root_halftone += '/'
            
        self.filenames = glob.glob(os.path.join(self.root_halftone,'*'+img_type))
        self.filenames_bases = []
        for _n in self.filenames :
            self.filenames_bases.append(os.path.splitext(os.path.basename(_n))[0])

        self.len = len(self.filenames)
        
    def __len__(self) :
        return self.len
    
    def __getitem__(self,idx) :
            
        # if no augmentation is used, directly use the image
        # generate both RGB and grayscale image
        imgh_name = self.root_halftone+self.filenames_bases[idx]+self.img_type
        imgH = cv2.imread(imgh_name,0)
        imgH = imgH.astype(np.float32)/255.0
        
        # add a dimension for batch processing
        imgH = np.expand_dims(imgH,0)
        
        # convert to tensors
        imgH = torch.from_numpy(imgH)
        
        # all the converted images are given as the data
        sample = {
            'halftone':imgH
        }
        
        return sample