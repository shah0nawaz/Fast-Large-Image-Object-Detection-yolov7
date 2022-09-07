import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from torch.utils.data import DataLoader
from torchvision import transforms

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.dataloader import LoadData

def run():
    device = 'cpu'
    weights = 'yolov7.pt'
    model = attempt_load(weights, map_location = device)
    stride = int(model.stride.max())
    imgsz = 608
    files_list = 'list.txt'
    
    #imgsz = check_img_size(imgsz, s = stride)
    infer_data = LoadData(files_list)
    
    dataset = DataLoader(infer_data, 
                         batch_size = 2,
                         num_workers = 4, 
                         pin_memory = True)
                         
    
    for img in dataset:
    
        print(img.shape)
        img = img.to(device)
        
        
        pred = model(img)[0]
        print(pred.shape)
        

run()  
                         
    
    
