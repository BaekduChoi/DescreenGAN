import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as SN
import torchvision

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

"""
    Code adapted from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
"""
class PerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, loss_type = 'mse', feature_layers=[3], before_act=False):
        super().__init__()

        assert loss_type in ['mse','l1']
        self.loss = F.l1_loss if loss_type == 'l1' else F.mse_loss

        # self.bl = (torchvision.models.vgg19(pretrained=True).features[:27].eval())
        # for p in self.bl.parameters():
        #     p.requires_grad = False

        blocks = []
        if before_act :
            blocks.append(torchvision.models.vgg19(pretrained=True).features[:3].eval())
            blocks.append(torchvision.models.vgg19(pretrained=True).features[3:8].eval())
            blocks.append(torchvision.models.vgg19(pretrained=True).features[8:17].eval())
            blocks.append(torchvision.models.vgg19(pretrained=True).features[17:26].eval())
        else :
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