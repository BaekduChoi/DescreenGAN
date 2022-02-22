import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as SN

def klvloss(mu,logvar) :
    return torch.mean(-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp()))

"""
    Loss for LSGAN
"""
class LSGANLoss(object) :
    def __init__(self,device) :
        super().__init__()
        self.loss = nn.MSELoss()
        self.device = device
        
    def get_label(self,prediction,is_real) :
        if is_real : 
            return torch.ones_like(prediction)
        else :
            return torch.zeros_like(prediction)
    
    def __call__(self,prediction,is_real) :
        label = self.get_label(prediction,is_real)
        label.to(self.device)
        return self.loss(prediction,label)

"""
    Loss for relativistic average LSGAN
"""
class RaLSGANLoss(object) :
    def __init__(self,device) :
        super().__init__()
        self.loss = nn.MSELoss()
        self.device = device
    
    def __call__(self,real,fake) :
        avg_real = torch.mean(real,dim=0,keepdim=True)
        avg_fake = torch.mean(fake,dim=0,keepdim=True)

        loss1 = self.loss(real-avg_fake,torch.ones_like(real).to(self.device))
        loss2 = self.loss(fake-avg_real,-torch.ones_like(fake).to(self.device))

        return loss1+loss2

"""
    Loss for hingeGAN discriminator
"""
class HingeGANLossD(object) :
    def __init__(self) :
        super().__init__()
    
    def __call__(self,prediction,is_real) :
        if is_real :
            loss = F.relu(1-prediction)
        else :
            loss = F.relu(1+prediction)

        return loss.mean()

"""
    Loss for hingeGAN generator
"""
class HingeGANLossG(object) :
    def __init__(self) :
        super().__init__()
    
    def __call__(self,prediction) :
        return -prediction.mean()

"""
    Loss for relativistic average hingeGAN
"""
class RaHingeGANLoss(object) :
    def __init__(self) :
        super().__init__()
    
    def __call__(self,real,fake) :
        avg_real = torch.mean(real,dim=0,keepdim=True)
        avg_fake = torch.mean(fake,dim=0,keepdim=True)

        dxr = real - avg_fake
        dxf = fake - avg_real

        loss1 = F.relu(1-dxr)
        loss2 = F.relu(1+dxf)

        return (loss1+loss2).mean()