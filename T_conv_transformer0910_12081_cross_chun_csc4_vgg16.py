import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from scipy import signal
from scipy import misc
import os
from torch.autograd import Variable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from cgg import VGGNet
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

__all__ = [
    'Conv_Trans_tiny', 'Conv_Trans_small', 'Conv_Trans_medium', 'Conv_Trans_large'
]

#convolutions define function
def conv3x3(in_planes, out_planes, stride=1, groups=1,dilation = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                 padding=1, groups=groups, bias=False,dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)#(1,32,64,64)
        return x

class double_conv1(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),#修改小波变换改这里
            # Downsample_LL(wavename='haar'),
            double_conv(in_ch, out_ch)
        )


    def forward(self, x):
        x1= self.mpconv(x)
        # LL1 = self.mpconv(x)
        return x1
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):

        x = self.conv(x)
        return x

class inconv1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv1, self).__init__()
        self.conv = double_conv1(in_ch, out_ch)

    def forward(self, x):

        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)


        return x

class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 bias=True,
                 norm_layer=nn.BatchNorm2d):
        """
        基本卷积模块，使用stride=2来替代pooling层
        :param in_channels: 输入通道数量
        :param out_channels: 输出通道
        :param kernel_size: 卷积核尺寸，默认为3
        :param stride: 步长，默认为2，用于替代pooling层
        :param padding: 默认为1
        :param bias: 是否使用 bias，默认使用
        :param norm_layer: bn 层类型，默认为 nn.BatchNorm2d，可以换成 apex.parallel.SyncBatchNorm
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               bias=bias,
                               padding=padding)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # GAN 的话建议使用 LeakyReLU
        # self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)

        return self.relu(self.bn(x))
class UNet(nn.Module):
    # def __init__(self, n_channels, n_classes, dgf_r, dgf_eps):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up1(1024, 256)
        self.up2 = up1(512, 128)
        self.up3 = up1(256, 64)
        self.up4 = up1(128, 64)
        self.outc = outconv(64, n_classes)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(64, 64)
    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def forward(self, x):
        x_0 =x
        x1 = self.inc(x_0)#shape([1,64,224,224])
        x2 = self.down1(x1)#LL shape([1,128,112,112]
        x3 = self.down2(x2)#LL shape([1,256,56,56])
        x4 = self.down3(x3)#LL shape([1,512,28,28])

        return x4


#计算多层感知机
class Mlp(nn.Module):#支持向量机的计算。三层，全连接网络，一个隐藏层，两个恢复层
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)#避免弥散
        x = self.fc2(x)
        x = self.drop(x)
        return x

#计算多头注意力，自注意力计算
class Attention_Chan(nn.Module):#自注意的计算
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):#自定义的头数为8
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."#确定duotouzhuyili分几个tou

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads#每一个注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5#gongshixiabiannayibufen

        self.v = nn.Linear(dim, dim, bias=qkv_bias)#类似与cnn中的卷积操作#resplace to convolution
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)#resplace to convolution
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) #position projection
        self.proj_drop = nn.Dropout(proj_drop) #position projection

        self.sr_ratio = sr_ratio#空间降低算子的比率
        if sr_ratio > 1:#这里用于计算空间降低注意力块
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape#(1,16384,64)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)#计算得到q#(1,1,16384,64)

        if self.sr_ratio > 1:#计算k,v。这里的k和v是通过空间降低注意力得到的。
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)#对张量重新排列维度，并改变形状。(1,64,128,128)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)#(1,256,64)
            x_ = self.norm(x_)#对x降低维度
            qk = self.qk(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#k=K*x,q=Q*x,计算得到看，v(2,1,1,256,64)
        else:
            qk = self.qk(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#不进行空间维度降低，常规方法生成的自注意力键值
        q, k = qk[0], qk[1]#(1,1,256,64)

        attn = (q.transpose(-2, -1) @ k) * self.scale#计算q*k'，在进行一个归一化处理#(1,1,64,64)
        attn = attn.softmax(dim=-1)#使用softmax激活
        attn = self.attn_drop(attn)#避免弥散

        x = (v @ attn ).transpose(1, 2).reshape(B, N, C)#计算与v相乘(1,16384,64),
        x = self.proj(x)#做一次线性操作(1,16384,64),
        x = self.proj_drop(x)#避免弥散
        return x
class Attention_spc(nn.Module):#自注意的计算
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):#自定义的头数为8
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."#确定duotouzhuyili分几个tou

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads#每一个注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5#gongshixiabiannayibufen

        self.q = nn.Linear(dim, dim, bias=qkv_bias)#类似与cnn中的卷积操作#resplace to convolution
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)#resplace to convolution
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) #position projection
        self.proj_drop = nn.Dropout(proj_drop) #position projection
        self.attn_C = Attention_Chan(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.sr_ratio = sr_ratio#空间降低算子的比率
        if sr_ratio > 1:#这里用于计算空间降低注意力块
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x1 = x
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)#计算得到q#(1,8,1024,16)

        if self.sr_ratio > 1:#计算k,v。这里的k和v是通过空间降低注意力得到的。
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)#对张量重新排列维度，并改变形状。
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)#对x降低维度
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#k=K*x,q=Q*x,计算得到看，v
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#不进行空间维度降低，常规方法生成的自注意力键值
        k, v = kv[0], kv[1]#(1,8,256,32)

        attn = (q @ k.transpose(-2, -1)) * self.scale#计算q*k'，在进行一个归一化处理(1,8,256,256)
        attn = attn.softmax(dim=-1)#使用softmax激活(1,8,256,256)
        attn = self.attn_drop(attn)#避免弥散(1,8,256,256)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)#计算与v相乘(1,16384,64),(1,256,256)
        x = self.proj(x)#做一次线性操作(1,16384,64),(1,256,256)
        x = self.proj_drop(x)#避免弥散(1,256,256)
        x2 = x1 + x #(1,256,256)
        x2_ = self.norm(x2)
        x2_ = self.attn_C(x2_,H,W)
        x2_ = x2 + x2_
        return x2_
class Attention_spa1(nn.Module):#自注意的计算
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):#自定义的头数为8
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."#确定duotouzhuyili分几个tou

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads#每一个注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5#gongshixiabiannayibufen

        self.q = nn.Linear(dim, dim, bias=qkv_bias)#类似与cnn中的卷积操作#resplace to convolution
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)#resplace to convolution
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) #position projection
        self.proj_drop = nn.Dropout(proj_drop) #position projection

        self.sr_ratio = sr_ratio#空间降低算子的比率
        if sr_ratio > 1:#这里用于计算空间降低注意力块
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)#计算得到q(1,8,16384,4)(1,8,4096,8)(1,8,1024,16)(1,8,256,32)

        if self.sr_ratio > 1:#计算k,v。这里的k和v是通过空间降低注意力得到的。
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)#对张量重新排列维度，并改变形状。(1,32,128,128)(1,64,64,64)(1,128,32,32)(1,256,16,16)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)#(1,256,32)(1,64,64)(1,16,128)(1,4,256)
            x_ = self.norm(x_)#对x降低维度#(1,256,32)(1,64,64)(1,16,128)(1,4,256)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#k=K*x,q=Q*x,计算得到看，v
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#不进行空间维度降低，常规方法生成的自注意力键值
        k, v = kv[0], kv[1]#(1,8,256,4)(1,8,64,8)(1,8,16,16)(1,8,4,32)

        attn = (q @ k.transpose(-2, -1)) * self.scale#计算q*k'，在进行一个归一化处理(1,8,16384,256)(1,8,4096,64)(1,8,1024,16)(1,8,256,4)
        attn = attn.softmax(dim=-1)#使用softmax激活(1,8,16384,256)(1,8,4096,64)(1,8,1024,16)(1,8,256,4)
        attn = self.attn_drop(attn)#避免弥散(1,8,16384,256)(1,8,4096,64)(1,8,1024,16)(1,8,256,4)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)#计算与v相乘(1,16384,32),(1,4096,64),(1,1024,128)(1,256,256)
        x = self.proj(x)#做一次线性操作(1,16384,32),(1,4096,64)(1,1024,128)(1,256,256)
        x = self.proj_drop(x)#避免弥散(1,16384,32)(1,4096,64)(1,1024,128)(1,256,256)

        return x
class Attention_spa2(nn.Module):#自注意的计算
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):#自定义的头数为8
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."#确定duotouzhuyili分几个tou

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads#每一个注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5#gongshixiabiannayibufen

        self.q = nn.Linear(dim, dim, bias=qkv_bias)#类似与cnn中的卷积操作#resplace to convolution
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)#resplace to convolution
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) #position projection
        self.proj_drop = nn.Dropout(proj_drop) #position projection

        self.sr_ratio = sr_ratio#空间降低算子的比率
        if sr_ratio > 1:#这里用于计算空间降低注意力块
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)#计算得到q(1,8,16384,4)(1,8,4096,8)(1,8,1024,16)(1,8,256,32)

        if self.sr_ratio > 1:#计算k,v。这里的k和v是通过空间降低注意力得到的。
            y_ = y.permute(0, 2, 1).reshape(B, C, H, W)#对张量重新排列维度，并改变形状。(1,32,128,128)(1,64,64,64)(1,128,32,32)(1,256,16,16)
            y_ = self.sr(y_).reshape(B, C, -1).permute(0, 2, 1)#(1,256,32)(1,64,64)(1,16,128)(1,4,256)
            y_ = self.norm(y_)#对x降低维度#(1,256,32)(1,64,64)(1,16,128)(1,4,256)
            kv = self.kv(y_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#k=K*x,q=Q*x,计算得到看，v
        else:
            kv = self.kv(y).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#不进行空间维度降低，常规方法生成的自注意力键值
        k, v = kv[0], kv[1]#(1,8,256,4)(1,8,64,8)(1,8,16,16)(1,8,4,32)

        attn = (q @ k.transpose(-2, -1)) * self.scale#计算q*k'，在进行一个归一化处理(1,8,16384,256)(1,8,4096,64)(1,8,1024,16)(1,8,256,4)
        attn = attn.softmax(dim=-1)#使用softmax激活(1,8,16384,256)(1,8,4096,64)(1,8,1024,16)(1,8,256,4)
        attn = self.attn_drop(attn)#避免弥散(1,8,16384,256)(1,8,4096,64)(1,8,1024,16)(1,8,256,4)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)#计算与v相乘(1,16384,32),(1,4096,64),(1,1024,128)(1,256,256)
        x = self.proj(x)#做一次线性操作(1,16384,32),(1,4096,64)(1,1024,128)(1,256,256)
        x = self.proj_drop(x)#避免弥散(1,16384,32)(1,4096,64)(1,1024,128)(1,256,256)

        return x
#transformer一个编码块
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_spc(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):#(1,16384,32),(1,4096,64)(1,1024,128)(1,256,256)(1,1024,128)(1,4096,64)(1,16384,32)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))#(1,16384,32),(1,4096,64),(1,1024,128)(1,256,256)(1,1024,128)(1,4096,64),(1,16384,32),
        x = x + self.drop_path(self.mlp(self.norm2(x)))#(1,16384,32),(1,4096,64),(1,1024,128)(1,256,256)(1,1024,128)(1,4096,64),(1,16384,32),
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1):
        """
        特征融合模块 FFM，这里 in_channels 为要融合的特征的通道数之和
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param reduction: attention 模块的通道数衰减
        """
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                               nn.Conv2d(out_channels,
                                                         out_channels // reduction,
                                                         kernel_size=1,
                                                         bias=False),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(out_channels // reduction,
                                                         out_channels,
                                                         kernel_size=1,
                                                         bias=False),
                                               nn.Sigmoid())# 计算weight map

    def forward(self, x1):
        fm_se = self.channel_attention(x1)# 计算weight map
        output = x1 * fm_se

        return output

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 5, 7), 'kernel size must be 3, 5, 7'
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.max =nn.MaxPool2d(1)
        self.conv1x1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print(f'avg:{avg_out.shape}')#([1, 1, 448, 448])
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print(f'max:{max_out.shape}')#([1, 1, 448, 448])
        scale = torch.cat([avg_out, max_out], dim=1)
        # print(f'sca:{scale.shape}')#([1, 2, 448, 448])
        scale = self.conv1(scale)
        # print(f'sca:{scale.shape}')#([1, 1, 448, 448])
        scale = self.sigmoid(scale)
        # print(f'sca:{scale.shape}')#([1, 1, 448, 448])
        x = x * scale
        x = self.conv1x1(x)

        return x
class ASPP(nn.Module):
    # have bias and relu, no bn
    def __init__(self, in_channel=2, out_channels=512, depth=32):
        super().__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Sequential(nn.Conv2d(2, depth, 1, 1), nn.ReLU(inplace=True))

        self.atrous_block1 = nn.Sequential(nn.Conv2d(2,depth, 1, 1),
                                           nn.ReLU(inplace=True))
        self.atrous_block6 = nn.Sequential(nn.Conv2d(2, depth, 3, 1, padding=6, dilation=6),
                                           nn.ReLU(inplace=True))
        self.atrous_block12 = nn.Sequential(nn.Conv2d(2, depth, 3, 1, padding=12, dilation=12),
                                            nn.ReLU(inplace=True))
        self.atrous_block18 = nn.Sequential(nn.Conv2d(2, depth, 3, 1, padding=18, dilation=18),
                                            nn.ReLU(inplace=True))

        self.conv_1x1_output = nn.Sequential(nn.Conv2d(depth * 5, 2, 1, 1), nn.ReLU(inplace=True))

        self.chan = ChannelAttention(in_channel, out_channels)
        self.spaa = SpatialAttention(in_channel, out_channels)

    def forward(self, x):#(16,2,128,128)(32,2,64,64)(64,2,32,32)
        size = x.shape[2:]#(128,128)(64,64)(32,32)(16,16)

        image_features = self.mean(x)#(16,2,1,1)(32,2,1,1)(64,2,1,1)(64,4,1,1)
        image_features = self.conv(image_features)#(16,32,1,1)(32,64,1,1)(64,128,1,1)
        image_features = F.upsample(image_features, size=size, mode='bilinear', align_corners=True)#(16,32,128,128)(32,64,64,64)(64,128,32,32)

        atrous_block1 = self.atrous_block1(x)#(16,32,128,128)(32,64,64,64)(64,128,32,32)
        atrous_block1 = self.spaa(atrous_block1)#(16,32,128,128)(32,64,64,64)(64,128,32,32)
        atrous_block1 = self.chan(atrous_block1)#(32,64,64,64)(32,64,64,64)(64,128,32,32)

        atrous_block6 = self.atrous_block6(x)#(16,32,128,128)(32,64,64,64)(64,128,32,32)
        atrous_block6 = self.spaa(atrous_block6)#(16,32,128,128)(32,64,64,64)(64,128,32,32)
        atrous_block6 = self.chan(atrous_block6)#(16,32,128,128)(32,64,64,64)(64,128,32,32)

        atrous_block12 = self.atrous_block12(x)#(16,32,128,128)(32,64,64,64)(64,128,32,32)
        atrous_block12 = self.spaa(atrous_block12)#(16,32,128,128)(32,64,64,64)(64,128,32,32)
        atrous_block12 = self.chan(atrous_block12)#(16,32,128,128)(32,64,64,64)(64,128,32,32)

        atrous_block18 = self.atrous_block18(x)#(16,32,128,128)(32,64,64,64)(64,128,32,32)
        atrous_block18 = self.spaa(atrous_block18)#(16,32,128,128)(32,64,64,64)(64,128,32,32)
        atrous_block18 = self.chan(atrous_block18)#(16,32,128,128)(32,64,64,64)(64,128,32,32)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))#(32,2,64,64)(64,2,32,32)
        return net

class SC_layer(nn.Module):
    '''
    Constructs an attention module for the convolutional layer
    '''

    def __init__(self,channel, depth, groups=32,):
        super(SC_layer,self).__init__()
        self.groups = groups
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))  # 组归一化
        self.sp = ASPP(channel,channel,depth)

    @staticmethod
    def channel_shuffle(x, groups):  # 通道混淆实现
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)  # 改变输入的通道数，分组。比如原来是1组，分成三组。(_1,2,128,112,112)
        x = x.permute(0, 2, 1, 3,
                      4)  # permute,对任意高维矩阵进行转置，调用方式只能是tensor.permute（）。而transpose只能实现对二维矩阵的转置。(1,128,2,112,112)
        # 可连续使用transpose实现高维矩阵的转置。
        # flatten
        x = x.reshape(b, -1, h, w)  # 恢复通道混淆后的形状。(1,256,112,112)

        return x

    def forward(self,x):
        b, c, h, w = x.shape  # (1,32,128,128)(1,64,64,64)(1,132,32,32)#(1,256,16,16)
        x = x.reshape(b * self.groups, -1, h, w) #(16,2,128,128) (32,2,64,64)(64,2,32,32)(64,4,16,16)# 改变形状，将组分离为子特征的形式，其中-1代表的是n，n为张量的长度除以第一个参数(64,4,112,112)
        x = self.sp(x)#(16,2,128,128)(32,2,64,64)(64,2,32,32)
        x = x.reshape(b,-1,h,w)# (1,32,128,128)(1,64,64,64)(1,128,32,32)
        x = self.channel_shuffle(x,2) #(1,32,128,128)(1,64,64,64)(1,128,32,32)
        return x


class AFC(nn.Module):
    def __init__(self, features, M, G, r, dim, patch_size, stride=1, L=32, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.#branch的数量
            G: num of convolution groups.#卷机组的数量
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(AFC, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."  # 确定duotouzhuyili分几个tou

        self.dim = dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每一个注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5  # gongshixiabiannayibufen

        self.q = nn.Linear(dim, dim, bias=qkv_bias)  # 类似与cnn中的卷积操作#resplace to convolution
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  # resplace to convolution
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # position projection
        self.proj_drop = nn.Dropout(proj_drop)  # position projection
        self.attn_S = Attention_spa2(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)

        d = max(int(features / r), L)  # 文中计算（4）这里用于实现自适应调整
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        self.M = M
        self.features = features  # 输入通道的维度
        self.conv = nn.Conv2d(features, features, 1, 1)  # 1*1conv
        self.conv1 = nn.Conv2d(features*2, features, 1, 1)  # 1*1conv
        self.conv2 = nn.Conv2d(features, features, 1, 1)  # 1*1conv
        self.layer1 = nn.Sequential(nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1))
        self.fc = nn.Linear(features, d)  # 第一次全连接层
        self.fcs = nn.ModuleList([])  # 第二次全连接层
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        self.chan = ChannelAttention(features, features)
        self.proj = nn.Conv2d(features, features, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(features)

    def forward(self, x1,x2):
        B, C, H, W = x1.shape#(1,32,128,128)(1,64,64,64)(1,128,32,32)(1,256,16,16)

        # fea_1 = self.conv(x1)  # (1,32,128,128)(1,64,64,64)(1,128,32,32)(1,256,16,16)
        # fea_2 = self.conv(x2)  # (1,32,128,128)(1,64,64,64)(1,128,32,32)(1,256,16,16)
        # x2_2 = self.proj(fea_2).flatten(2).transpose(1, 2)
        # x2_2 = self.norm(x2_2)
        # # fea_U = torch.cat([fea_1, fea_2],dim=1)#(1,32,128,128)(1,64,64,64)(1,128,32,32)(1,256,16,16)
        # # fea_U= self.conv1(fea_U)
        # fea_U = self.proj(fea_1).flatten(2).transpose(1, 2)  # (1,16384,32)(1,4096,64)(1,1024,128)(1,256,256)
        # fea_U = self.norm(fea_U)  # (1
        # fea_U = self.attn_S(fea_U, x2_2 ,H, W)#(1,16384,32)(1,4096,64)(1,1024,128)(1,256,256)
        # fea_U = fea_U.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()#(1,32,128,128)(1,64,64,64)(1,128,32,32)(1,256,16,16)
        # fea_4 = self.conv2(fea_U)  # (1,32,128,128)(1,64,64,64)(1,128,32,32)(1,256,16,16)
        fea_4 = self.chan(x1)

        return fea_4
#图像块的嵌入
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
      convolution shixian
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):#一共每个图片长和宽分成16*16的图像块，则一张图长可以变为14个，宽可以变为14个，每一小块线性变换之后的维度为768
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        num_patches = self.H * self.W#256
        self.num_patches = num_patches #16*16  256
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        B, C, H, W = x.shape#(1,256,16,16)

        x = self.proj(x).flatten(2).transpose(1, 2)#一个位置嵌入过程，位置嵌入中的线性化self.proj(x)wei (1,64,128,128)#(1,256,128)
        x = self.norm(x)#块嵌入中的而归一化#(1,16384,64)(1,256,128)
        H, W = H // self.patch_size[0], W // self.patch_size[1]#16//1

        return x, (H, W)
class PatchEmbed1(nn.Module):
    """ Image to Patch Embedding1
      convolution shixian decoder
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):#一共每个图片长和宽分成16*16的图像块，则一张图长可以变为14个，宽可以变为14个，每一小块线性变换之后的维度为768
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        num_patches = self.H * self.W#256
        self.num_patches = num_patches #16*16  256
        self.proj =  nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        B, C, H, W = x.shape#(1,256,16,16)(1,128,32,32)(1,64,64,64)(1,32,128,128)

        x = self.proj(x).flatten(2).transpose(1, 2)#一个位置嵌入过程，位置嵌入中的线性化self.proj(x)wei (1,64,128,128)#(1,4096,64)(1,16384,32)(1,65536,3)
        x = self.norm(x)#块嵌入中的而归一化#(1,16384,64)(1,4096,64)(1,16384,32)(1,65536,3)
        H, W = H // self.patch_size[0], W // self.patch_size[1]#32,32,64,64128,128

        return x, (H, W)
class PatchEmbed2(nn.Module):
    """ Image to Patch Embedding1
      convolution shixian decoder
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):#一共每个图片长和宽分成16*16的图像块，则一张图长可以变为14个，宽可以变为14个，每一小块线性变换之后的维度为768
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        num_patches = self.H * self.W#256
        self.num_patches = num_patches #16*16  256
        self.proj =  nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        B, C, H, W = x.shape#(1,256,16,16)(1,128,32,32)(1,64,64,64)(1,32,128,128)

        x = self.proj(x).flatten(2).transpose(1, 2)#一个位置嵌入过程，位置嵌入中的线性化self.proj(x)wei (1,64,128,128)#(1,4096,64)(1,16384,32)(1,65536,3)
        x = self.norm(x)#块嵌入中的而归一化#(1,16384,64)(1,4096,64)(1,16384,32)(1,65536,3)
        H, W = H // self.patch_size[0], W // self.patch_size[1]#32,32,64,64128,128

        return x, (H, W)
class up1(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):#256,64
        super(up1, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch , in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
class up2(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):#256,64
        super(up2, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2 , in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
#金字塔视觉变换用于分ge任务
class PyramidVision_Conv_Transformer(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=3, num=3,  num_classes=4, embed_dims=[32, 64, 128, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):#depths是每一个阶段的transformer block数
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num = num
        self.vgg_model = VGGNet(requires_grad = True, remove_fc = True)
        # patch_embed块嵌入
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0])#512,128

        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])#128,64

        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])#64,32

        self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2],
                                       embed_dim=embed_dims[3])#32,16
        # decoder_patch_embed块嵌入
        self.patch_embed_1 = PatchEmbed1(img_size=img_size//16, patch_size=1, in_chans=embed_dims[3],
                                       embed_dim=embed_dims[2])#32

        self.patch_embed_2 = PatchEmbed1(img_size=img_size//8, patch_size=1, in_chans=embed_dims[2],
                                       embed_dim=embed_dims[1])#16

        self.patch_embed_3 = PatchEmbed1(img_size=img_size//4, patch_size=1, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[0])#64

        self.patch_embed_4 = PatchEmbed2(img_size=img_size//1, patch_size=1, in_chans=embed_dims[0],
                                       embed_dim=num_classes)#128

        self.c1 = outconv(64,32)
        self.c2 = outconv(128, 64)
        self.c3 = outconv(256, 128)
        self.c4 = outconv(512, 256)
        self.inc = inconv(num,32)
        self.inc1 = inconv1(num,32)
        self.down1 = down(32,64)
        self.down2 = down(64,128)
        self.down3 = down(128,256)
        self.conv2 = nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1)
        self.gn2 = nn.GroupNorm(64,64)
        self.SC1 = SC_layer(32,32,16)
        self.SC2 = SC_layer(64,64,32)
        self.SC3 = SC_layer(128,128,64)
        self.SC4 = SC_layer(256,256,128)
        self.SC5 = SC_layer(128,128,64)
        self.SC6 = SC_layer(64,64,32)
        self.SC7 = SC_layer(32,32,16)
        self.conv1 = double_conv(256, 128)
        self.conv2 = double_conv(128, 64)
        self.conv3 = double_conv(64, 32)
        self.conv4 = double_conv(6, 4)
        self.up1 = up1(256,128)
        self.up2 = up1(128,64)
        self.up3 = up1(64, 32)
        self.up4 =nn.ConvTranspose2d(32,4, kernel_size=4, stride=4)


        # pos_embed#位置嵌入#suijishengchengdelingjuzhen
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))#(1,16384,64)
        self.pos_drop1 = nn.Dropout(p=drop_rate)#第一个通道维度阶段
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, embed_dims[1]))#(1,4096,128)
        self.pos_drop2 = nn.Dropout(p=drop_rate)#第二个通道维度阶段
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, embed_dims[2]))#(1,1024,320)
        self.pos_drop3 = nn.Dropout(p=drop_rate)#第三个通道维度阶段
        self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed4.num_patches, embed_dims[3]))#(1,256,512)
        self.pos_drop4 = nn.Dropout(p=drop_rate)#第四个通道维度阶段

        #decoder_pos_embed#位置嵌入#suijishengchengdelingjuzhen
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.patch_embed_1.num_patches, embed_dims[2]))  # (1,1024,128)
        self.pos_drop_1 = nn.Dropout(p=drop_rate)  # 第一个通道维度阶段
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, self.patch_embed_2.num_patches, embed_dims[1]))  # (1,4096,128)
        self.pos_drop_2 = nn.Dropout(p=drop_rate)  # 第二个通道维度阶段
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, self.patch_embed_3.num_patches, embed_dims[0]))  # (1,1024,320)
        self.pos_drop_3 = nn.Dropout(p=drop_rate)  # 第三个通道维度阶段
        self.pos_embed_4 = nn.Parameter(torch.zeros(1, self.patch_embed_4.num_patches , num_classes))  # (1,257,512)
        self.pos_drop_4 = nn.Dropout(p=drop_rate)  # 第四个通道维度阶段
        self.pos_embed_5 = nn.Parameter(torch.zeros(1, self.patch_embed_4.num_patches, num_classes))  # (1,257,512)
        self.pos_drop_5 = nn.Dropout(p=drop_rate)  # 第四个通道维度阶段

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule随机深度衰减规则，
                                                                                  # 返回一个1维张量，包含在区间start和end上均匀间隔的step个点。
        cur = 0   #update=2
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])#第一阶段块
        self.AFC1 = AFC(32, 2, 8, 2, 32, 1, stride=1, L=32, num_heads=8, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop_rate, proj_drop=dpr[2], sr_ratio=sr_ratios[0])

        cur += depths[0]#cur=2
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])#第二个阶段快
        self.AFC2 = AFC(64, 2, 8, 2, 64, 1, stride=1, L=32, num_heads=8, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop_rate, proj_drop=dpr[4], sr_ratio=sr_ratios[1])

        cur += depths[1]#cur=4
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])#第三个阶段块
        self.AFC3 = AFC(128, 2, 8, 2, 128, 1, stride=1, L=32, num_heads=8, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop_rate, proj_drop=dpr[6], sr_ratio=sr_ratios[2])

        cur += depths[2]#cur=6
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])#第四个阶段块
        self.AFC4 = AFC(256, 2, 8, 2, 256, 1, stride=1, L=32, num_heads=8, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop_rate, proj_drop=dpr[7], sr_ratio=sr_ratios[3])

        self.norm = norm_layer(embed_dims[3])#最后一个阶段做一个层归一化
       #decoder
        cur_1 = 0  # update=2
        self.block_1 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_1 + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])  # 第一阶段块
        self.AFC5 = AFC(128, 2, 8, 2, 128, 1, stride=1, L=32, num_heads=8, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop_rate, proj_drop=dpr[7], sr_ratio=sr_ratios[3])


        cur_1 += depths[3]  # cur=2
        self.block_2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_1 + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])  # 第二个阶段快
        self.AFC6 = AFC(64, 2, 8, 2, 64, 1, stride=1, L=32, num_heads=8, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop_rate, proj_drop=dpr[6], sr_ratio=sr_ratios[2])


        cur_1 += depths[2]  # cur=4
        self.block_3 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_1 + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])  # 第三个阶段块
        self.AFC7 = AFC(32, 2, 8, 2, 32, 1, stride=1, L=32, num_heads=8, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop_rate, proj_drop=dpr[4], sr_ratio=sr_ratios[1])


        cur_1 += depths[1]  # cur=6
        self.block_4 = nn.ModuleList([Block(
            dim=num_classes, num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_1 + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])  # 第四个阶段块


        self.norm = norm_layer(embed_dims[3])  # 最后一个阶段做一个层归一化
        self.norm1 = norm_layer(num)
        # # cls_token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))#类嵌入，PVT中将其嵌入到最后一个阶段，在vit中是每一个阶段都有(1,1,512)
        #
        # # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()#分类头，做一个线性表示

        # init weights初始化模型的参数
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)
        trunc_normal_(self.pos_embed_1, std=.02)
        trunc_normal_(self.pos_embed_2, std=.02)
        trunc_normal_(self.pos_embed_3, std=.02)
        trunc_normal_(self.pos_embed_4, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)#初始化权重

    def reset_drop_path(self, drop_path_rate):#重新设置drop，动态的设置了一个更新的drop_path参数
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))] #stochastic depth decay rule随机深度衰减规则，
                                                                                  # 返回一个1维张量，包含在区间start和end上均匀间隔的step个点。
        cur = 0
        for i in range(self.depths[0]):#第一阶段
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):#第二阶段
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):#第三阶段
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):#第四阶段
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def _init_weights(self, m):#初始化权重的定义
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):#重新设置分类
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x1, x2):#(1,3,512,512)
        B = x1.shape[0]#B为#B=1

        low_level_feature = self.vgg_model(x2)
        c41 = low_level_feature['x4']
        c4 = self.c4(c41)
        c31 = low_level_feature['x3']
        c3 = self.c3(c31)
        c21 = low_level_feature['x2']
        c2 = self.c2(c21)
        c11 = low_level_feature['x1']
        c1= self.c1(c11)

        # stage 1
        x1, (H, W) = self.patch_embed1(x1)#阶段一的块嵌入(1,16384,32),(128,128)
        x1 = x1 + self.pos_embed1#阶段一的块嵌入与juedui位置嵌入
        x1 = self.pos_drop1(x1)#位置嵌入以后做一个drop_out执行一个优化(1,16384,32,
        for blk in self.block1:
            x1 = blk(x1, H, W)#(1,16384,32),
        x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()#(1,32,128,128)
        # x2_a = self.inc1(x2)#(1,32,128,128)
        x2_a = c1
        x2_b = self.SC1(x2_a)#(1,32,128,128)
        x_stage1_1 = x1 + x2_b #xia yi ceng trans de shu ru
        x_1_1 = self.AFC1(x1,x2_a) # (1,32,128,128)
        x_stage1_2 = x2_a + x_1_1 #xia yi ceng juan ji de shu ru
        # x_1 = torch.cat([x1, x_1], dim=1)
        # x_1 = self.conv3(x_1)

        # stage 2第二阶段的transformer编码
        x_1_, (H, W) = self.patch_embed2(x_stage1_1)#(1,4096,64)
        # x2 = self.pos_embed2
        x_1_ = x_1_ + self.pos_embed2#(1,4096,64)
        x_1_ = self.pos_drop2(x_1_)#(1,4096,64)
        for blk in self.block2:
            x_1_ = blk(x_1_, H, W)#(1,4096,64)
        x_1_ = x_1_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()#(1,64,64,64)

        # x_2 = self.down1(x_stage1_2)#(1,64,64,64)
        x_2 = c2
        x_2_c = self.SC2(x_2)#(1,64,64,64)
        x_stage2_1 = x_1_ + x_2_c  #xia yi ceng trans de shu ru
        x_2_1 = self.AFC2(x_1_,x_2)#(1,64,64,64)
        x_stage2_2 = x_2 + x_2_1 #xia yi ceng juan ji de shu ru

        # x_2_ = torch.cat([x_1_, x_2_], dim=1)
        # x_2_ = self.conv2(x_2_)

        # stage 3第三个阶段的transformer编码
        x_3, (H, W) = self.patch_embed3(x_stage2_1)#(1,1024,128)
        x_3 = x_3 + self.pos_embed3#(1,1024,128)
        x_3 = self.pos_drop3(x_3)#(1,1024,128)
        for blk in self.block3:
            x_3 = blk(x_3, H, W)#(1,1024,128)
        x_3 = x_3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()#(1,128,32,32)
        # x_3_a = self.down2(x_stage2_2)#(1,128,32,32)
        x_3_a = c3
        x_3_c = self.SC3(x_3_a)#(1,128,32,32)
        x_stage3_1 = x_3 + x_3_c   #xia yi ceng trans de shu ru

        x_3_1 = self.AFC3(x_3, x_3_a)#(1,128,32,32)
        x_stage3_2 = x_3_a + x_3_1   #xia yi ceng juan ji de shu ru
        # x_3_ = torch.cat([x_3, x_3_], dim=1)
        # x_3_ = self.conv1(x_3_)

        # stage 4第四个阶段的transformer编码，并在此阶段加入类编码
        x_4, (H, W) = self.patch_embed4(x_stage3_1)#(1,256,256)
        # cls_tokens = self.cls_token.expand(B, -1, -1)#类转换
        # x = torch.cat((cls_tokens, x), dim=1)#类与块的级联
        x_4 = x_4 + self.pos_embed4#最后的块嵌入与位置嵌入得到transformer block的输入数据(1,256,256)
        x_4 = self.pos_drop4(x_4)#(1,256,256)
        for blk in self.block4:
            x_4 = blk(x_4, H, W)#(1,256,512)
        x_4 = x_4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (1,256,16,16)
        # x_4_a = self.down3(x_stage3_2)# (1,256,16,16)
        x_4_a = c4
        x_4_c = self.SC4(x_4_a)# (1,256,16,16)
        x_stage4_1 = x_4 + x_4_c   #xia yi ceng trans de shu ru
        x_4_1 = self.AFC4(x_4, x_4_a)#(1,256,16,16)
        x_stage4_2 = x_4_a + x_4_1    #xia yi ceng juan ji de shu ru

        x_5, (H, W) = self.patch_embed_1(x_stage4_1)  # 阶段一的块嵌入(1,1024,128)))
        x_5 = x_5 + self.pos_embed_1  # 阶段一的块嵌入与juedui位置嵌入(1,1024,128)
        x_5 = self.pos_drop_1(x_5)  # 位置嵌入以后做一个drop_out执行一个优化(1,1024,128)
        for blk in self.block_1:
            x_5 = blk(x_5, H*2, W*2)  # (1,16384,64),
        x_5 = x_5.reshape(B, H*2, W*2, -1).permute(0, 3, 1, 2).contiguous()  # (1,128,32,32)
        x_5_a = self.up1(x_stage4_2, x_stage3_2)
        x_5_c = self.SC5(x_5_a)
        x_stage5_1 = x_5 + x_5_c  #xia yi ceng trans de shu ru
        x_5_stage_1 = torch.cat([x_stage5_1,x_stage3_1],dim=1)
        x_5_stage_1 = self.conv1(x_5_stage_1)#xia yi ceng trans de shu ru

        x_5_a1 = self.up1(x_stage4_1,x_stage3_1)
        x_5_1 = self.AFC5(x_5_a1, x_5_a)
        x_stage5_2 = x_5_a + x_5_1
        x_5_stage_2 = torch.cat([x_stage5_2, x_stage3_2], dim=1)
        x_5_stage_2 = self.conv1(x_5_stage_2) #xia yi ceng juan ji de shu ru

        # stage 2第二阶段的transformer编码
        x_6, (H, W) = self.patch_embed_2(x_5_stage_1)#(1,4096,64)
        # x2 = self.pos_embed2
        x_6 = x_6 + self.pos_embed_2#(1,4096,64)
        x_6 = self.pos_drop_2(x_6)#(1,4096,64)
        for blk in self.block_2:
            x_6 = blk(x_6, H*2, W*2)
        x_6 = x_6.reshape(B, H*2, W*2, -1).permute(0, 3, 1, 2).contiguous()#(1,64,64,64)
        x_6_a = self.up2(x_5_stage_2, x_stage2_2)
        x_6_c = self.SC6(x_6_a)
        x_stage6_1 = x_6 + x_6_c
        x_6_stage_1 = torch.cat([x_stage6_1, x_stage2_1], dim=1)
        x_6_stage6_1 = self.conv2(x_6_stage_1)  # xia yi ceng trans de shu ru

        x_6_a1 = self.up2(x_5_stage_1, x_stage2_1)
        x_6_1 = self.AFC6(x_6_a1, x_6_a)
        x_stage6_2 = x_6_a + x_6_1
        x_6_s_2 = torch.cat([x_stage6_2, x_stage2_2], dim=1)
        x_6_stage6_2 = self.conv2(x_6_s_2)    #xia yi ceng juan ji de shu ru

        # stage 3第三个阶段的transformer编码
        x_7, (H, W) = self.patch_embed_3(x_6_stage6_1)#(1,16384,32),64,64
        x_7 = x_7 + self.pos_embed_3#(1,16384,32)
        x_7 = self.pos_drop_3(x_7)#(1,16384,32)
        for blk in self.block_3:
            x_7 = blk(x_7, H*2, W*2)  # (1,1024,320)
        x_7 = x_7.reshape(B, H*2, W*2, -1).permute(0, 3, 1, 2).contiguous()  # (1,32,128,128)
        x_7_a = self.up3(x_6_stage6_2, x_stage1_2)
        x_7_c = self.SC7(x_7_a)
        x_stage7_1 = x_7 + x_7_c
        x_7_stage_1 = torch.cat([x_stage7_1, x_stage1_1], dim=1)
        x_7_stage7_1 = self.conv3(x_7_stage_1)  # xia yi ceng trans de shu ru

        x_7_a1 = self.up3(x_6_stage6_1, x_stage1_1)
        x_7_1 = self.AFC7(x_7_a1,x_7_a)
        x_stage7_2 = x_7_a + x_7_1
        x_7_s_2 = torch.cat([x_stage7_2, x_stage1_2], dim=1)
        x_7_stage7_2 = self.conv3(x_7_s_2)     #xia yi ceng juan ji de shu ru

        # stage 4第四个阶段的transformer编码，并在此阶段加入类编码
        x_8, (H, W) = self.patch_embed_4(x_7_stage7_1)#(1,65536,3)
        # cls_tokens = self.cls_token.expand(B, -1, -1)#类转换
        # x = torch.cat((cls_tokens, x), dim=1)#类与块的级联
        x_8 = x_8 + self.pos_embed_4  # 最后的块嵌入与位置嵌入得到transformer block的输入数据(1,65536,3)
        x_8 = self.pos_drop_4(x_8)#(1,65536,3)
        for blk in self.block_4:
            x_8 = blk(x_8, H*4, W*4)  # (1,65536,3)
        x_8 = x_8.reshape(B, H*4, W*4, -1).permute(0, 3, 1, 2).contiguous()  # (1,3,512,512)

        return x_8

    def forward(self, x1, x2):
        x = self.forward_features(x1 , x2)#执行所有编码阶段获得x，在此基础上执行分类步骤#(1,512)


        return x


def _conv_filter(state_dict, patch_size=16):#转换手动+线性块嵌入为卷积过程
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def pvt_tiny(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_small(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_medium(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_large(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_huge_v2(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[128, 256, 512, 768], num_heads=[2, 4, 8, 12], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 10, 60, 3], sr_ratios=[8, 4, 2, 1],
        # drop_rate=0.0, drop_path_rate=0.02)
        **kwargs)
    model.default_cfg = _cfg()

    return model


if __name__=='__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bs = 1
    data1 = torch.randn(bs, 3, 256, 256).to(device)
    data2 = torch.randn(bs, 3, 128, 128).to(device)
    for norm_layer in [nn.BatchNorm2d]:
        model = PyramidVision_Conv_Transformer( patch_size=4, embed_dims=[32, 64, 128, 256], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1]).to(device)
        a = model(data1, data2)
        print(f'y:{a.shape}')#[1,1000]

