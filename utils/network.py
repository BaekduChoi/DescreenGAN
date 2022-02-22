import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

from torch import nn
import torch
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.utils import spectral_norm as SN
from misc import default_init_weights

class ResConvBlock(nn.Module): 
    def __init__(self,ndf,ksize) :
        super().__init__()
        pad = ksize//2
        self.conv1 = nn.Conv2d(ndf,ndf,ksize,1,pad)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(ndf,ndf,ksize,1,pad)
    
    def forward(self,x) :
        return x + self.conv2(self.act(self.conv1(x)))

"""
    Instead of using conv-conv-relu as in the paper, we use conv-relu-conv-relu
    conv-conv is inherently ambiguous
"""
class DSBlock(nn.Module) :
    def __init__(self,ndf,ksize) :
        super().__init__()
        pad = ksize//2
        self.ds = nn.Conv2d(ndf,ndf*2,ksize,2,pad)
        self.conv = nn.Conv2d(ndf*2,ndf*2,ksize,1,pad)
        self.act = nn.ReLU()
    
    def forward(self,x) :
        return self.act(self.conv(self.act(self.ds(x))))

class USBlockRes(nn.Module) :
    def __init__(self,ndf,ksize) :
        super().__init__()
        pad = ksize//2
        self.us = nn.ConvTranspose2d(ndf*2,ndf,2,2)
        self.conv = nn.Conv2d(ndf,ndf,ksize,1,pad)
        self.act = nn.ReLU()
    
    def forward(self,x,x_prev=None) :
        return self.act(self.conv(self.act(self.us(x))))+x_prev

"""
    Content aggregation net
"""
class ContentNet(nn.Module) :
    def __init__(self) :
        super().__init__()
        ksize = 3
        ndf = 64
        pad = ksize//2
        self.inblock = nn.Sequential(
            nn.Conv2d(1,ndf,ksize,1,pad),
            nn.ReLU(),
            ResConvBlock(ndf,ksize)
        )
        self.dsblock1 = DSBlock(ndf,ksize)
        self.dsblock2 = DSBlock(ndf*2,ksize)

        self.br = nn.Sequential(
            ResConvBlock(ndf*4,ksize),
            ResConvBlock(ndf*4,ksize),
            ResConvBlock(ndf*4,ksize),
            ResConvBlock(ndf*4,ksize)
        )

        self.usblock1 = USBlockRes(ndf*2,ksize)
        self.usblock2 = USBlockRes(ndf,ksize)

        self.outblock = nn.Sequential(
            ResConvBlock(ndf,ksize),
            nn.Conv2d(ndf,1,1)
        )
    
    def forward(self,x) :
        x0 = self.inblock(x)
        x1 = self.dsblock1(x0)
        x = self.br(self.dsblock2(x1))
        x = self.usblock1(x,x1)
        x = self.usblock2(x,x0)
        return self.outblock(x)

class DetailNet(nn.Module) :
    def __init__(self,in_ch=2) :
        super().__init__()
        ksize = 3
        ndf = 64
        pad = ksize//2

        L = [nn.Conv2d(in_ch,ndf,ksize,1,pad),nn.ReLU()]
        for _ in range(8) :
            L.append(ResConvBlock(ndf,ksize))
        L.append(nn.Conv2d(ndf,1,1))

        self.block = nn.Sequential(*L)
    
    def forward(self,hft,context) :
        return context + self.block(torch.cat([hft,context],dim=1))

class PRLnet(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.content = ContentNet()
        self.detail = DetailNet()
    
    def forward(self,x) :
        x1 = self.content(x)
        x2 = self.detail(x,x1)

        return x1,x2

class ConvBlock(nn.Module): 
    def __init__(self,in_ch,out_ch,ksize) :
        super().__init__()
        pad = ksize//2
        self.conv1 = nn.Conv2d(in_ch,out_ch,ksize,1,pad)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch,out_ch,ksize,1,pad)
        self.bn2 = nn.BatchNorm2d(out_ch)
    
    def forward(self,x) :
        return self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))))

class USBlock(nn.Module) :
    def __init__(self,in_ch,out_ch,ksize) :
        super().__init__()
        pad = ksize//2
        self.us = nn.ConvTranspose2d(in_ch,out_ch,2,2)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv = nn.Conv2d(out_ch,out_ch,ksize,1,pad)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
    
    def forward(self,x,x_prev=None):
        x_in = torch.cat([x,x_prev],dim=1) if x_prev != None else x
        return self.act(self.bn2(self.conv(self.act(self.bn1(self.us(x_in))))))

"""
    Discriminator for GAN-based implementation
    Based on the PatchGAN structure in https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    Added Dropout for slower discriminator fitting
"""
class Discriminator(nn.Module) :
    def __init__(self,in_ch=1,ndf=64,bn=True,sn=False) :
        super().__init__()
        
        if bn :
            L = [
                nn.Conv2d(in_ch,ndf,kernel_size=4,stride=2,padding=1,padding_mode='circular'),
                nn.LeakyReLU(0.2,True),
                nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
                nn.BatchNorm2d(ndf*2),
                nn.LeakyReLU(0.2,True),
                nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
                nn.BatchNorm2d(ndf*4),
                nn.LeakyReLU(0.2,True),
                nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
                nn.BatchNorm2d(ndf*8),
                nn.LeakyReLU(0.2,True),
                nn.Conv2d(ndf*8,ndf*8,kernel_size=4,stride=1,padding=1,bias=False,padding_mode='circular'),
                nn.BatchNorm2d(ndf*8),
                nn.LeakyReLU(0.2,True),
                nn.Conv2d(ndf*8,1,kernel_size=4,stride=1,padding=1,padding_mode='circular')
            ]
        else :
            L = [
                nn.Conv2d(in_ch,ndf,kernel_size=4,stride=2,padding=1,padding_mode='circular'),
                nn.LeakyReLU(0.2,True),
                nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
                nn.LeakyReLU(0.2,True),
                nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
                nn.LeakyReLU(0.2,True),
                nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
                nn.LeakyReLU(0.2,True),
                nn.Conv2d(ndf*8,ndf*8,kernel_size=4,stride=1,padding=1,bias=False,padding_mode='circular'),
                nn.LeakyReLU(0.2,True),
                nn.Conv2d(ndf*8,1,kernel_size=4,stride=1,padding=1,padding_mode='circular')
            ]
        
        if sn :
            for i in range(len(L)) :
                if isinstance(L[i],nn.Conv2d) :
                    L[i] = SN(L[i])
        
        self.block = nn.Sequential(*L)
    
    def forward(self,x) :
        return self.block(x)

"""
    RRDB in ESRGAN
"""
class RDB(nn.Module) :
    def __init__(self,ndf) :
        super().__init__()

        self.conv1 = nn.Conv2d(ndf,ndf,3,1,1)
        self.conv2 = nn.Conv2d(2*ndf,ndf,3,1,1)
        self.conv3 = nn.Conv2d(3*ndf,ndf,3,1,1)
        self.conv4 = nn.Conv2d(4*ndf,ndf,3,1,1)
        self.conv5 = nn.Conv2d(5*ndf,ndf,3,1,1)

        self.act = nn.LeakyReLU(0.2)

        self.beta = 0.2

        default_init_weights([self.conv1,self.conv2,self.conv3,self.conv4,self.conv5],0.1)
    
    def forward(self,x) :

        x1 = self.conv1(x)
        x1 = self.act(x1)
        x2 = self.conv2(torch.cat([x,x1],dim=1))
        x2 = self.act(x2)
        x3 = self.conv3(torch.cat([x,x1,x2],dim=1))
        x3 = self.act(x3)
        x4 = self.conv4(torch.cat([x,x1,x2,x3],dim=1))
        x4 = self.act(x4)

        return self.act(self.conv5(torch.cat([x,x1,x2,x3,x4],dim=1)))*self.beta+x

class RRDB(nn.Module) :
    def __init__(self,ndf) :
        super().__init__()

        self.rdb1 = RDB(ndf)
        self.rdb2 = RDB(ndf)
        self.rdb3 = RDB(ndf)

        self.beta = 0.2

    def forward(self,x) :
        x1 = self.rdb1(x)
        x1 = self.rdb2(x1)
        x1 = self.rdb3(x1)

        return x+self.beta*x1

class DescreenNet(nn.Module) :
    def __init__(self,ndf=32) :
        super().__init__()
        self.inblk = nn.Sequential(
            nn.Conv2d(1,ndf,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf,ndf,3,1,1)
        )
        
        num_blks_mid = 12
        L = [RRDB(ndf) for i in range(num_blks_mid)]

        self.mid = nn.Sequential(*L)

        self.out = nn.Conv2d(ndf//4,1,1)
    
    def forward(self,x) :
        x = self.inblk(x)
        x = self.mid(F.interpolate(x,scale_factor=0.5,mode='bilinear'))
        return self.out(F.pixel_shuffle(x,2))

class Unet(nn.Module) :
    def __init__(self,in_ch=1,sn=True) :
        super().__init__()

        ndf = 8

        L_in = [SN(nn.Conv2d(in_ch,ndf,3,1,1))] if sn else [nn.Conv2d(in_ch,ndf,3,1,1)]
        L_in += [nn.LeakyReLU(0.2)]
        self.inblk = nn.Sequential(*L_in)

        L1 = [SN(nn.Conv2d(ndf,ndf,3,1,1))] if sn else [nn.Conv2d(ndf,ndf,3,1,1)]
        L1 += [nn.LeakyReLU(0.2)]
        L1 += [SN(nn.Conv2d(ndf,2*ndf,3,2,1))] if sn else [nn.Conv2d(ndf,2*ndf,3,2,1)]
        L1 += [nn.LeakyReLU(0.2)]
        self.ds1 = nn.Sequential(*L1)

        L2 = [SN(nn.Conv2d(2*ndf,2*ndf,3,1,1))] if sn else [nn.Conv2d(2*ndf,2*ndf,3,1,1)]
        L2 += [nn.LeakyReLU(0.2)]
        L2 += [SN(nn.Conv2d(2*ndf,4*ndf,3,2,1))] if sn else [nn.Conv2d(2*ndf,4*ndf,3,2,1)]
        L2 += [nn.LeakyReLU(0.2)]
        self.ds2 = nn.Sequential(*L2)

        L3 = [SN(nn.Conv2d(4*ndf,4*ndf,3,1,1))] if sn else [nn.Conv2d(4*ndf,4*ndf,3,1,1)]
        L3 += [nn.LeakyReLU(0.2)]
        L3 += [SN(nn.Conv2d(4*ndf,8*ndf,3,2,1))] if sn else [nn.Conv2d(4*ndf,8*ndf,3,2,1)]
        L3 += [nn.LeakyReLU(0.2)]
        self.ds3 = nn.Sequential(*L3)

        L_mid = []
        for _ in range(4) :
            L_mid += [SN(nn.Conv2d(8*ndf,8*ndf,3,1,1))] if sn else [nn.Conv2d(8*ndf,8*ndf,3,1,1)]
            L_mid += [nn.LeakyReLU(0.2)]
        self.mid = nn.Sequential(*L_mid)

        L4 = [SN(nn.Conv2d(8*ndf,8*ndf,3,1,1))] if sn else [nn.Conv2d(8*ndf,8*ndf,3,1,1)]
        L4 += [nn.LeakyReLU(0.2)]
        L4 += [SN(nn.ConvTranspose2d(8*ndf,4*ndf,2,2))] if sn else [nn.ConvTranspose2d(8*ndf,4*ndf,2,2)]
        L4 += [nn.LeakyReLU(0.2)]
        self.us1 = nn.Sequential(*L4)

        L5 = [SN(nn.Conv2d(8*ndf,4*ndf,3,1,1))] if sn else [nn.Conv2d(8*ndf,4*ndf,3,1,1)]
        L5 += [nn.LeakyReLU(0.2)]
        L5 += [SN(nn.ConvTranspose2d(4*ndf,2*ndf,2,2))] if sn else [nn.ConvTranspose2d(4*ndf,2*ndf,2,2)]
        L5 += [nn.LeakyReLU(0.2)]
        self.us2 = nn.Sequential(*L5)

        L6 = [SN(nn.Conv2d(4*ndf,2*ndf,3,1,1))] if sn else [nn.Conv2d(4*ndf,2*ndf,3,1,1)]
        L6 += [nn.LeakyReLU(0.2)]
        L6 += [SN(nn.ConvTranspose2d(2*ndf,ndf,2,2))] if sn else [nn.ConvTranspose2d(2*ndf,ndf,2,2)]
        L6 += [nn.LeakyReLU(0.2)]
        self.us3 = nn.Sequential(*L6)

        L_out = [SN(nn.Conv2d(2*ndf,ndf,3,1,1))] if sn else [nn.Conv2d(2*ndf,ndf,3,1,1)]
        L_out += [nn.LeakyReLU(0.2)]
        L_out += [SN(nn.Conv2d(ndf,1,1))] if sn else [nn.Conv2d(ndf,1,1)]
        self.outblk = nn.Sequential(*L_out)
    
    def forward(self,x) :
        x = self.inblk(x)
        x1 = self.ds1(x)
        x2 = self.ds2(x1)
        x3 = self.ds3(x2)
        x3 = self.mid(x3)
        x3 = self.us1(x3)
        x2 = self.us2(torch.cat([x3,x2],dim=1))
        x1 = self.us3(torch.cat([x2,x1],dim=1))
        
        return self.outblk(torch.cat([x1,x],dim=1))