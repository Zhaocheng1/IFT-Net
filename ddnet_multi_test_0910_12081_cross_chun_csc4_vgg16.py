import os
import torch
import cv2
import torch.nn as nn
from torchvision import transforms,models
import torch.nn.functional as F
import numpy as np
from T_conv_transformer0910_12081_cross_chun_csc4_vgg16 import PyramidVision_Conv_Transformer
from T_model.Unet import UNet1
from functools import partial


os.environ['CUDA_VISIBLE_DEVICES']='2'

def conv3x3(in_planes, out_planes, stride=1, groups=1,dilation = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                 padding=1, groups=groups, bias=False,dilation=dilation)
class ddnet(nn.Module):
    def __init__(self,inchannel):
        super(ddnet, self).__init__()
        num_category = 4
        groups =1

        self.f_transformer = PyramidVision_Conv_Transformer(img_size=256, patch_size=4, in_chans=3, num=3,  num_classes=4, embed_dims=[32, 64, 128, 256], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1])#224*224
        self.gn1 = nn.GroupNorm(3, 3)
        self.conv2 = conv3x3(inchannel,inchannel, 2, groups)  # stride moved
        self.conv3 = conv3x3(num_category, num_category, 1, groups)  # stride moved



    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
    def forward(self, x):

        x1 = F.relu(self.gn1(self.conv2(x)))#suoxiaodechicun
        _, _, h, w = x1.size()
        # print(f'x1:{x1.shape}')
        fea1=self.f_transformer(x,x1)
        # print(f'fea1:{fea1.shape}')
        out_1=self.conv3(fea1)
        return out_1

if __name__=='__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bs = 1
    data1 = torch.randn(bs, 3, 256, 256).to(device)
    for norm_layer in [nn.BatchNorm2d]:
        model = ddnet(3).to(device)
        a = model(data1)
        print(f'y:{a.shape}')#[1,1000]