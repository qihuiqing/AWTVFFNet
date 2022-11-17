# coding: utf-8
'''
Created on 2022-03-03 12:48:42
Title: common
Copyright: Copyright (c) 2021 
School: ECNU
author: Chyi  
version 1.0  
'''
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
class add_noise:
    def __init__(self,args, img_data):
        self.noise_level = args.noise_level
        self.img = img_data
        self.noise_type = args.noise_style
        if args.noise_style == 'Gaussian':
            self.img_noise = self.get_noise()
            
        if args.noise_style == 'Gaussian+salt+pepper':
            self.img_noise = self.get_mixture_01()
        
        if args.noise_style == 'Gaussian+speckle':
            self.img_noise = self.get_mixture_02()
        
        if args.noise_style == 'salt+pepper+speckle':
            self.img_noise = self.get_mixture_03()
        
    def get_noise(self):
        if self.noise_type=='Gaussian':
            sigma = self.noise_level/255
            noise = torch.tensor(np.random.normal(scale = sigma,size = self.img.shape))
            img_noise = torch.clamp(self.img+noise,0,1)
        return img_noise
    
    def get_mixture_01(self): 
        if self.noise_type=='Gaussian+salt+pepper':
            prob = 0.005
            sigma = self.noise_level/255
            noise = torch.tensor(np.random.normal(scale = sigma,size = self.img.shape))
            g = torch.tensor(np.random.random(size=self.img.shape))
            b=self.img
            b[g<prob/2]=0
            b[(g>=prob/2) & (g<prob)]=1.0
            img_noise = torch.clamp(self.img+noise+b,0,1)
        return img_noise
    
    def get_mixture_02(self): 
        if self.noise_type=='Gaussian+speckle':
            prob = 0.005
            sigma = self.noise_level/255
            noise = torch.tensor(np.random.normal(scale = sigma,size = self.img.shape))
            g = torch.tensor(np.random.random(size=self.img.shape))
            g = np.sqrt(12*prob)*(g-0.5)
            img_noise = torch.clamp(self.img+noise+self.img*g,0,1)
        return img_noise
    
    
    def get_mixture_03(self):
        if self.noise_type=='salt+pepper+speckle':
            prob = 0.005
            g = torch.tensor(np.random.random(size=self.img.shape))
            b=self.img
            b[g<prob/2]=0
            b[(g>=prob/2) & (g<prob)]=1.0
            g = np.sqrt(12*prob)*(g-0.5)
            img_noise = torch.clamp(self.img+self.img*g+b,0,1)
        return img_noise
    
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


def plot_image_grid(images_np, save_path, nrow =2, factor=1, interpolation='lanczos'):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = torchvision.utils.make_grid(images_np, nrow)
    grid = grid.numpy()
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    
    plt.savefig(save_path)
    plt.close()
    
    return grid


def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    torch_grid = torchvision.utils.make_grid(images_np, nrow)
    
    return torch_grid.numpy()