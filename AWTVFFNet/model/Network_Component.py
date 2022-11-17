# coding: utf-8
'''
Created on 2022-7-06
Title: AWTVFF_Net 
Copyright: Copyright (c) 2021 
School: ECNU
author: Chyi  
version 1.0  
'''
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Tuple
from PIL import Image

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=1, stride=1, bias=True, activation='lrelu', norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, bias=bias)
        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out

class skip_block(nn.Module):
    #This is the general skip connection block
    def __init__(self, in_size, out_size, kernel_size, stride=1):
        super(skip_block,self).__init__()
        convs = [nn.ReflectionPad2d(kernel_size//2), ConvBlock(in_size, out_size, kernel_size =kernel_size,stride = stride )]
        self.convs = nn.Sequential(*convs)

    def forward(self,x):
        return self.convs(x)


class Downsample_block(nn.Module):
    #This is the down sampling basic block.
    def __init__(self, in_size, out_size, kernel_size, stride):
        super(Downsample_block,self).__init__()
        convs = [nn.ReflectionPad2d(kernel_size//2), ConvBlock(in_size, out_size, kernel_size =kernel_size, stride = stride ),
                      nn.ReflectionPad2d(kernel_size//2), ConvBlock(out_size, out_size, kernel_size =kernel_size, stride = 1)
            ]

        self.down = nn.Sequential(*convs)

    def forward(self,x):   
        return self.down(x)

class Upsample_block(nn.Module):
    # This is the up sampling basic block
    def __init__(self, in_size, out_size, kernel_size, stride =1):
        super(Upsample_block,self).__init__()
        convs = [nn.BatchNorm2d(in_size),
                nn.ReflectionPad2d(kernel_size//2), ConvBlock(in_size, out_size, kernel_size =kernel_size,stride = stride ),
                nn.ReflectionPad2d(0), ConvBlock(out_size, out_size, kernel_size =1,stride = stride )
            ]
        self.up_convs = nn.Sequential(*convs)
    
    def forward(self, x):
        return self.up_convs(x)


class AuEncoder(nn.Module):
    #this is the auto-encoder for extract features
    def __init__(self, in_size, out_size, kernel_size=3, stride=2):
        super(AuEncoder, self).__init__()
        convs = [ConvBlock(in_size, 16, kernel_size =kernel_size, stride = 1, activation='relu', norm='batch'),
                ConvBlock(16, 32, kernel_size =kernel_size, stride = stride, activation='relu', norm='batch'),
                ConvBlock(32, 64, kernel_size =kernel_size, stride = stride, activation='relu', norm='batch'),
                ConvBlock(64, out_size, kernel_size =kernel_size, stride = stride, activation='relu', norm='batch'),
                ConvBlock(out_size, out_size, kernel_size =kernel_size, stride = stride, activation='relu', norm='batch'),
                ConvBlock(out_size, out_size, kernel_size =kernel_size, stride = stride, activation='relu', norm='batch'),
            ]
        self.AE = nn.Sequential(*convs)
    
    def forward(self,x):
        return self.AE(x)

class baisc_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(baisc_conv, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        out = self.conv2d(self.reflection_pad(x))
        return out

class ResidualBlock(nn.Module):# ResidualBlocks
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = baisc_conv(channels, channels, kernel_size=3, stride=1)
        self.conv2 = baisc_conv(channels, channels, kernel_size=3, stride=1)
        self.relu =nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv2(self.conv1(x))) * 0.1
        out = torch.add(out, residual)
        return out


# class ResidualBlock(nn.Module):# ResidualBlocks  
#     def __init__(self, channels,out_channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = baisc_conv(channels, channels, kernel_size=3, stride=1)
#         self.conv2 = baisc_conv(channels, channels, kernel_size=3, stride=1)
#         self.relu = nn.ReLU()
#         self.conv3 = baisc_conv(channels, out_channels, kernel_size=3, stride=1)

#     def forward(self, x):
#         residual = x
#         out = self.conv2(self.relu(self.conv1(x))) * 0.1
#         out = self.conv3(torch.add(out, residual))
#         return out

class RBG(nn.Module): # ResidualBlocks group
    def __init__(self, channels, numbers):
        super(RBG, self).__init__()
        convs = []
        for _ in range(numbers):
            convs.append(ResidualBlock(channels))
        
        self.rbg =  nn.Sequential(*convs)
        
    def forward(self, x):
        return self.rbg(x)

# class RBG(nn.Module): # ResidualBlocks group 
#     def __init__(self, channels, out_channels, numbers):
#         super(RBG, self).__init__()
#         convs = [ResidualBlock(channels, out_channels)]
#         for _ in range(numbers-1):
#             convs.append(ResidualBlock(out_channels,out_channels))
        
#         self.rbg =  nn.Sequential(*convs)
        
#     def forward(self, x):
#         return self.rbg(x)


class Last_block(nn.Module):
    # This is the last layer for the skip output
    def __init__(self,in_size, out_size, kernel_size, stride =1):
        super(Last_block, self).__init__()
        convs = [nn.ReflectionPad2d(kernel_size//2),
                 nn.Conv2d(in_size, out_size, kernel_size= kernel_size, stride= stride)]
        self.convs = nn.Sequential(*convs)
        self.last = nn.Sigmoid()
        
    def forward(self, x):
        return self.last(self.convs(x))


def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        input_image image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input_image image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img

def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf

def D(x, Dh_DFT, Dv_DFT):
    x_DFT = torch.fft.fft2(x, dim=(-2,-1)).cuda("cuda:3")#cuda index
    Dh_x = torch.fft.ifft2(Dh_DFT*x_DFT, dim=(-2,-1)).real
    Dv_x = torch.fft.ifft2(Dv_DFT*x_DFT, dim=(-2,-1)).real
    return Dh_x, Dv_x


def norm2_loss(x):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    vec = torch.pow(x[:,:,:,:], 2)
    
    return torch.sum(vec)


def Weight_computed(D_h, D_v, alpha=1):
    #compute the k
    D_norm = torch.sqrt(torch.pow(D_h,2) + torch.pow(D_v,2))
    D_h_m = D_h - torch.ones_like(D_h)*torch.mean(D_norm, dim =(2,3), keepdim= True)
    D_h_m = torch.ones_like(D_h)*torch.mean(D_h_m, dim =(2,3), keepdim= True)
    D_v_m = D_v - torch.ones_like(D_v)*torch.mean(D_norm, dim =(2,3), keepdim= True)
    D_v_m = torch.ones_like(D_v)*torch.mean(D_v_m, dim =(2,3), keepdim= True)
    
    k = torch.ones_like(D_h)* torch.mean(D_h_m + D_v_m, dim =(2,3), keepdim= True)
    k = 1.5 *k

    #computing the new weight
    div = torch.ones_like(D_h) + torch.pow(D_norm,2) * torch.exp(2*torch.pow(torch.div(D_norm, k),2))
    weight = torch.div(1, div)

    return weight + alpha

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]
