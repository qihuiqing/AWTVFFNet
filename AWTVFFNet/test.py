# coding: utf-8
'''
Created on 2022-09-05 15:15:07
Title: test  
Copyright: Copyright (c) 2021 
School: ECNU
author: Chyi  
version 1.0  
'''
import os
import argparse
import data
import torch
from torchvision.utils import save_image
from model.myuitls import *
from model import AWTVF2Net 
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument('--dir_data', type = str, default = 'test_data', help ='data path')
parser.add_argument('--data_name', type = str, default = 'UCL_N', help ='dataset name')
parser.add_argument('--save_path', type = str, default = 'results', help ='the path of results saving')
parser.add_argument('--model_path', type = str, default = 'train_result/G+S/final_net_max.pt', help ='the path of results saving')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--train_only', action='store_false', help='only train')
parser.add_argument('--test_only', action='store_false', help='only test')
parser.add_argument('--batch_size', type = int, default = 1, help='batch size')
parser.add_argument('--re_size', type = int, default = 256 , help='resize')
parser.add_argument('--noise_level', type = int , default = 15,
                     help='noise level')
parser.add_argument('--kernel_size', type = int , default = 3 ,
                     help='kernel_size')
parser.add_argument('--skip_size', type = int , default = 4 ,
                     help='skip connection size')
parser.add_argument('--down_num_channel', type = int , default = 128 ,
                     help='the down sampling chanel numbers')
parser.add_argument('--mode', type = str , default = 'bilinear' ,
                     help='the mode of upsample')
parser.add_argument('--growRate', type = int , default = 64,
                     help='the growing rate for Residual blocks')

parser.add_argument('--n_feats', type = int , default = 64,
                     help='number of features')

parser.add_argument('--n_blocks', type = int , default = 4,
                     help='number of blocks for RBS')
parser.add_argument('--n_layer', type = int , default = 3,
                     help='number of layers for EBS')

args = parser.parse_args()
test_loader = data.TestDatatload(args).test_loader
device_index = "cuda:3"
net = AWTVF2Net.make_net(args).to(device_index)
net.load_state_dict(torch.load(args.model_path))
print(len(test_loader))
for i,(img,label) in enumerate(test_loader):
    
    img_noise = img
    img_noise = img_noise.type(dtype).to(device_index)
    denoised, netout = net(img_noise)
    out = denoised.detach().to(torch.device('cpu'))
    save_path = os.path.join(args.save_path,args.data_name, str(args.noise_level))
    os.makedirs(save_path, exist_ok=True)
    file_name = os.path.join(save_path, label[0]+'denoised.png')
    #save image
    save_image(out,file_name)
    print("sucessful!")
