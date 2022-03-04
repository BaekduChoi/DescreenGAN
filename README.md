# Photorealistic descreening of halftone images using conditional generative adversarial networks (tentative title)

Baekdu Choi (choi504@purdue.edu) and J. P. Allebach

This is an implementation of photorealistic descreening using cGANs in [PyTorch](https://pytorch.org/). 

VGG-based perceptual loss code is borrowed from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49.

The training images are generated using [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) as the original images. The images are first divided into 256x256 patches and then [DBS](https://ieeexplore.ieee.org/document/877215) is performed on them.

To train :

1) Train without GAN using train_cRaGAN_nongan.py and train_cRaGAN_nongan.json
2) Using the result from 1), train with GAN using train_cRaGAN.py and train_cRaGAN.json