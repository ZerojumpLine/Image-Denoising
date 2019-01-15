# Image Denoising
## Overview
The denoising method was developed based on EDSR [Paper](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf), an advanced super-resolution method. I found it could outperform other network architecture such as DnCNN, U-net, Densenet and RCAN in the case of image denoising.
I used this method to beat other 40+ teams to win the competition.

## Task
The introduction of the challege is here (http://eucompetition.huawei.com/uk/). Organizers provided the GT images and noisy images (They added gaussian noise). We were asked to reduce the noise and recover the original images.

## Method
I briefly summarized the details of the method in slide[PDF] (https://github.com/ZerojumpLine/Image-Denoising/blob/master/Huawei_Denoise_ZejuLi.pdf). Basically, all the code which is needed to reproduce my results is uploaded to this repository.

## Acknowledgement
This code borrows heavily from [pytorch-RCAN-and-EDSR](https://github.com/yulunzhang/RCAN) and [DnCNN](https://github.com/cszn/DnCNN).
