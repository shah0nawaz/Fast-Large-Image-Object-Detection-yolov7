import glob
import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision import transforms 

class CustomImageDataset(Dataset):
    def __init__(self, files_list, transform = None):
        self.img_dir = pd.read_csv(files_list)
        self.transform = transform
        
        
    def extract_name(self, path):
        name = path.split('/')[-1].split('.')[0]
        return name
        
        
    def __len__(self):
        return len(self.img_dir)
        
    def __getitem__(self, idx):
        path = self.img_dir.iloc[idx,0]
        #print(self.path)
        
        img = cv2.imread(path)
        name = self.extract_name(path)
        #img = cv2.resize(img0, (416,416))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        #print(type(img))
        img = img/255.0
        #print(type(img))
        
        return img, name


if __name__== '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    customdataset = LoadImages('list.txt')
    infer_data = DataLoader(customdataset, 
                            batch_size = 64,
                            num_workers = 4,
                            pin_memory = True)
    #print(infer_data.batch_size)
    for img, name in infer_data:
        print(len(name))
        img.to(device)
        print(img.shape)
        print(img.type)
       
    
        



