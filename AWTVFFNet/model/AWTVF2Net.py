# coding: utf-8
'''
Created on 2022-08-16 21:55:12
Title: AWTVF2Net 
Copyright: Copyright (c) 2022 
School: ECNU
author: Chyi  
version 1.0  
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .myuitls import *

def make_net(args):
    return AWTVFF(args)


class EBS(nn.Module):
    # This is the feature extracting blocks set 
    def __init__(self, args, n_layer=4): #n_layer=8
        super(EBS, self).__init__()
        n_feats = args.n_feats
        rate  = args.growRate
        kernel_size = args.kernel_size #3
        
        convs = [Extract_Block(n_feats, rate, kernel_size)]
        for n in range(1, n_layer):
            convs.append(Extract_Block(n_feats + n * rate, rate, kernel_size))
        self.convs = nn.Sequential(*convs)
        self.last = nn.Sequential(*[nn.Conv2d(n_feats*2 + n_layer * rate, n_feats, 1, padding=0, stride=1), bn(n_feats), act('ReLU')])
    
    def forward(self, x):
        out = self.last(torch.cat((x,self.convs(x)), dim=1))
        return out


class RBS(nn.Module):
    #This is the residual blocks set.
    def __init__(self, args, conv=basic_conv):
        super(RBS, self).__init__()
        n_feats = args.n_feats
        kernel_size = args.kernel_size#3
        n_resblock = args.n_blocks #5
        res_block_set = [ResBlock(conv, n_feats, kernel_size) for _ in range(n_resblock)]
        self.rbs_conv = nn.Sequential(*res_block_set)
        self.last = nn.Sequential(*[nn.Conv2d(n_feats*2, n_feats, 1, padding=0, stride=1), bn(n_feats), act('ReLU')])

    def forward(self, x):
        out = self.last(torch.cat((x, self.rbs_conv(x)), dim=1))
        return out
    
##main module 1    
class SOSB(nn.Module): #need adjust
    def __init__(self, args):
        super(SOSB, self).__init__()
        skip_size = args.skip_size
        kernel_size = args.kernel_size
        
        self.up = nn.ConvTranspose2d(skip_size, skip_size, kernel_size, padding=1, stride=2)
        self.fu =  nn.Sequential(*[nn.Conv2d(skip_size, skip_size, 1, padding=0, stride=1), bn(skip_size), act()])
    
    def forward(self, x, y): 
        out = self.up(x,y.size())#up_scale
        out = torch.add(out,y)
        out = self.fu(out)+y
        
        return out

##main module 2 
class AuEncoder(nn.Module):
    #this is the auto-encoder for extract features
    def __init__(self, args, input_size, output_size, conv = basic_conv):
        super(AuEncoder, self).__init__()
        n_layer = args.n_layer
        n_feats = args.n_feats
        kernel_size = args.kernel_size
        self.AU = nn.Sequential(*[conv(input_size, output_size, kernel_size), bn(n_feats), act('ReLU'), EBS(args, n_layer), RBS(args)])

    def forward(self, x):
        return self.AU(x)


##main module 3
class FB(nn.Module):
    def __init__(self, args):
        super(FB, self).__init__()
        n_feats = args.n_feats
        self.fb = nn.Sequential(*[nn.Conv2d(n_feats * 2, n_feats, 1, padding=0, stride=1), bn(n_feats), act('ReLU'), RBS(args)])

    def forward(self, x):
        return self.fb(x)

class AWTVFF(nn.Module):
    def __init__(self, args, conv = basic_conv):
        super(AWTVFF,self).__init__()
        n_feats = args.n_feats
        kernel_size = args.kernel_size
        in_size = args.n_colors
        down_num_channel = args.down_num_channel
        skip_size = args.skip_size #4
        fuse_size = down_num_channel+skip_size
        self.mode = args.mode
        
        self.make_map1 = AuEncoder(args, in_size, n_feats)# for original noise image
        self.make_map2 = AuEncoder(args, skip_size, n_feats)# for sider out
        self.make_map3 = AuEncoder(args, in_size, n_feats)# for u_net out
        
        #u-net
        self.skip_01 = skip_block(in_size, skip_size, kernel_size = 1, stride = 1 )
        self.down_01 = Downsample_block(in_size, down_num_channel, kernel_size, stride = 2)
        self.skip_mid_01 = skip_block(down_num_channel, skip_size, kernel_size = 1, stride = 1)
        self.down_mid_01 = Downsample_block(down_num_channel, down_num_channel, kernel_size, stride = 2)
        self.skip_mid_02 = skip_block(down_num_channel, skip_size, kernel_size = 1, stride = 1)
        self.down_mid_02 = Downsample_block(down_num_channel, down_num_channel, kernel_size, stride = 2)
        self.skip_mid_03 = skip_block(down_num_channel, skip_size, kernel_size = 1, stride = 1)
        self.down_mid_03 = Downsample_block(down_num_channel, down_num_channel, kernel_size, stride = 2)
        self.skip_mid_04 = skip_block(down_num_channel, skip_size, kernel_size = 1, stride = 1)
        self.down_mid_04 = Downsample_block(down_num_channel, down_num_channel, kernel_size, stride = 2)
        
        self.up_mid_01 = Upsample_block(fuse_size, down_num_channel, kernel_size)
        self.up_mid_02 = Upsample_block(fuse_size, down_num_channel, kernel_size)
        self.up_mid_03 = Upsample_block(fuse_size, down_num_channel, kernel_size)
        self.up_mid_04 = Upsample_block(fuse_size, down_num_channel, kernel_size)
        self.up_mid_05 = Upsample_block(fuse_size, down_num_channel, kernel_size)
        self.last = Last_block(down_num_channel, in_size, kernel_size = 1)
        
        #sos boosting
        self.sosb1 = SOSB(args)
        self.sosb2 = SOSB(args)
        self.sosb3 = SOSB(args)
        self.sosb4 = SOSB(args)
        
        #fusing
        self.fu1 = FB(args)
        self.fu2 = FB(args)
        self.fu3 = FB(args)
        
        #out 
        self.tail = conv(n_feats, in_size, kernel_size)

    def forward(self, x):
        mode = self.mode
        f1 = self.make_map1(x)
        #u-net
        #encoder 
        skip0 = self.skip_01(x)#256
        resx = self.down_01(x)#128
        skip1 = self.skip_mid_01(resx)#128
        resx = self.down_mid_01(resx)#64
        skip2 = self.skip_mid_02(resx)#64
        resx = self.down_mid_02(resx)#32
        skip3 = self.skip_mid_03(resx)#32
        resx = self.down_mid_03(resx)#16
        skip4 = self.skip_mid_04(resx)#16
        resx = self.down_mid_04(resx)#8
        
        #SOS bosting
        out = self.sosb1(skip4,skip3)#32
        out = self.sosb2(out,skip2)#64
        out = self.sosb3(out,skip1)#128
        out = self.sosb4(out,skip0)#256
        f2 = self.make_map2(out)
        
        #decoder
        out = F.upsample(resx, skip4.size()[2:], mode = mode, align_corners=True)#16
        out = torch.cat([out, skip4], dim = 1)
        out = self.up_mid_01(out)#16
        out = F.upsample(out, skip3.size()[2:], mode = mode, align_corners=True)#32
        out = torch.cat([out, skip3], dim = 1)
        out = self.up_mid_02(out)#32
        out = F.upsample(out, skip2.size()[2:], mode = mode, align_corners=True)#64
        out = torch.cat([out, skip2], dim = 1)
        out = self.up_mid_03(out)#64
        out = F.upsample(out, skip1.size()[2:], mode = mode, align_corners=True)#128
        out = torch.cat([out, skip1], dim = 1)
        out = self.up_mid_04(out)#128
        out = F.upsample(out, skip0.size()[2:], mode = mode, align_corners=True)#256
        out = torch.cat([out, skip0], dim = 1)
        out = self.up_mid_05(out)#256
        out1 = self.last(out)
        f3 = self.make_map3(out1)

        #Fusing 
        out = torch.add(f1,f2)#1+2
        f1 = torch.add(f1,f3)#1+3
        f2 =torch.add(f2,f3)#2+3

        out = self.fu1(torch.cat((out,f1), dim=1))
        f1 = self.fu2(torch.cat((f1,f2), dim=1))
        
        out = self.fu3(torch.cat((out,f1), dim=1))
        out = self.tail(out)
        
        return out, out1