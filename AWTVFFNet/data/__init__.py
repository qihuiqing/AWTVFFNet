# coding: utf-8
'''
Created on 2022-03-03 16:04:59
Title: data  
Copyright: Copyright (c) 2021 
School: ECNU
author: Chyi  
version 1.0  
'''
import os
import glob
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image

class Datasetload:
    def __init__(self, args):
        super(Datasetload, self).__init__()
        self.train_loader = None #initila the train data
        self.dir_train = None
        if args.train_only:
            self.dir_train = self.set_file_path(args)
            trains = ATVDataset(args, self.dir_train)
            self.train_loader = DataLoader(dataset = trains, batch_size=args.batch_size, 
                                           shuffle=True, num_workers=args.num_workers)
        
    def set_file_path(self,args):
        if args.data_name == "DOTA":
            dir_train = os.path.join(args.dir_data,args.data_name, args.train_set)
        if args.data_name == "NWPU45":
            dir_train = os.path.join(args.dir_data,args.data_name)
        return dir_train


class TestDatatload:
    def __init__(self, args):
        super(TestDatatload, self).__init__()
        self.test_loader = None #initila the test data
        self.dir_test = None
        if args.test_only:
            self.dir_test = self.set_file_path(args)
            test = ATVDataset(args, self.dir_test)
            self.test_loader = DataLoader(dataset = test, batch_size=args.batch_size, 
                                           shuffle=True)
    def set_file_path(self,args):
        dir_test = os.path.join(args.dir_data,args.data_name, str(args.noise_level))
        return dir_test


class ATVDataset(Dataset):
    def __init__(self, args, root):
        super(ATVDataset, self).__init__()
        self.file_list = glob.glob(os.path.join(root,'*.*'))
        self.file_list = sorted(self.file_list, key=lambda x: x.split('.')[-2])
        if args.train_only:
            self.trains = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.trains = transforms.Compose([
                transforms.RandomCrop(args.re_size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                    
            ])
    
    def __getitem__(self, index):  
        """
        return image data and clean image
        """
        img_path = self.file_list[index]
        data = Image.open(img_path)
        data = self.trains(data)
        label = img_path.split("/")[-1]
        label = label[0:(len(label)-4)]
        return data,label         
    
    def __len__(self):
        """
        return images size.
        """
        return len(self.file_list)
        