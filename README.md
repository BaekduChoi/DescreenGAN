# Descreening of halftone images using generative adversarial networks

Baekdu Choi (choi504@purdue.edu) and J. P. Allebach
Electronic Imaging Systems Laboratory, Purdue University

This is an implementation of descreening using cGANs in [PyTorch](https://pytorch.org/). 

VGG-based perceptual loss code is borrowed from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49.

The training images are generated using [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) as the original images. The images are first divided into 256x256 patches and then [DBS](https://ieeexplore.ieee.org/document/877215) is performed on them.

Creating environment using Conda:
1) conda create -n myenv
2) conda activate myenv
3) conda install --file requirements.txt -c pytorch

To train:
1) Open train_cRaGAN_nongan.json and train_cRaGAN.json and modify the items as needed, especially the path to the dataset
2) Train without GAN using 'train_cRaGAN_nongan.py -opt train_cRaGAN_nongan.json' with arguments -nch and -blk (nch = number of channels in RRDBs, blk = number of RRDBs)
3) Using the result from 1), train with GAN using 'train_cRaGAN.py -opt train_cRaGAN.json' with arguments -nch and -blk

Example to test:
1) Open test_cRaGAN.json and modify the items as needed, especially the path to the test dataset.
2) test_cRaGAN.py -opt test_cRaGAN.json -nch 32 -blk 12
3) Note that -nch and -blk should match to what was used for training
