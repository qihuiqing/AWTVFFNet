# coding: utf-8
'''
Created on 2022-03-04 18:35:12
Title: utils  
Copyright: Copyright (c) 2021 
School: ECNU
author: Chyi  
version 1.0  
'''

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import numpy as np
from PIL import Image

def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'ReLU':
            return nn.ReLU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


class skip_block(nn.Module):
    #This is the general skip connection block
    def __init__(self, in_size, out_size, kernel_size, stride=1):
        super(skip_block,self).__init__()
        convs = [nn.ReflectionPad2d(kernel_size//2),
                 nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride),
                 bn(out_size),
                 act()]
        self.convs = nn.Sequential(*convs)

    def forward(self,x):
        out = self.convs(x)
        return out

class Downsample_block(nn.Module):
    #This is the down sampling basic block.
    def __init__(self, in_size, out_size, kernel_size, stride):
        super(Downsample_block,self).__init__()
        down_convs = [nn.ReflectionPad2d(kernel_size//2),
                      nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride),
                      bn(out_size),
                      act(),
                      nn.ReflectionPad2d(kernel_size//2),
                      nn.Conv2d(out_size, out_size, kernel_size=kernel_size, stride=1),
                      bn(out_size),
                      act()
            ]

        self.down = nn.Sequential(*down_convs)


    def forward(self,x):
        
        out = self.down(x)
        
        return out


class Upsample_block(nn.Module):
    # This is the up sampling basic block
    def __init__(self, in_size, out_size, kernel_size, stride =1):
        super(Upsample_block,self).__init__()
        convs = [bn(in_size),
                nn.ReflectionPad2d(kernel_size//2),
                nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride),
                bn(out_size),
                act(),
                nn.ReflectionPad2d(0),
                nn.Conv2d(out_size, out_size, kernel_size=1, stride=stride),
                bn(out_size),
                act()
                 ]
        self.up_convs = nn.Sequential(*convs)
    
    def forward(self, x):
        out = self.up_convs(x)
        
        return out
        
        
class Last_block(nn.Module):
    # This is the last layer for the skip output
    def __init__(self,in_size, out_size, kernel_size, stride =1):
        super(Last_block, self).__init__()
        convs = [nn.ReflectionPad2d(kernel_size//2),
                 nn.Conv2d(in_size, out_size, kernel_size= kernel_size, stride= stride)]
        
        self.convs = nn.Sequential(*convs)
        self.last = nn.Sigmoid()
        
    def forward(self, x):
        out = self.last(self.convs(x))
        return out
        

def basic_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class Extract_Block(nn.Module):
    #This is the basic block for extracting the feature maps
    def __init__(self, inChannels, growRate, kernel_size = 3):
        super(Extract_Block, self).__init__()
        n_feats = inChannels
        rate  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(n_feats, rate, kernel_size, padding=(kernel_size-1)//2, stride=1),
            bn(rate),
            act('ReLU'),
            nn.Conv2d(rate, rate, kernel_size, padding=(kernel_size-1)//2, stride=1),
            bn(rate),
            act('ReLU')
        ])
    
    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)
        
class ResBlock(nn.Module): 
    #The is the basic Residual block
    def __init__(self, conv, n_feats, kernel_size, bias=True):
        super(ResBlock, self).__init__()
        self.convs = nn.Sequential(*[conv(n_feats, n_feats, kernel_size, bias = bias),
                                     bn(n_feats),
                                     act('ReLU')])
        self.last =nn.Sequential(*[nn.Conv2d(n_feats*2, n_feats, kernel_size = 3, padding=1, stride = 1), act('ReLU')]) 
    def forward(self, x):
        out = self.convs(x)
        out = self.last(torch.cat((x, out), dim = 1))
        return out

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
    x_DFT = torch.fft.fft2(x, dim=(-2,-1)).cuda("cuda:1")#cuda index
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