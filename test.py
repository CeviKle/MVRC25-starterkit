from cProfile import label
from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np
import os
import cv2
import pandas as pd
from tqdm import tqdm
import cv2 as cv
from torchvision import models
import os
from restoreNet import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model_r = torch.load('latest_2_r.pth').to(device)

def norm(x):
    return (x-x.min())/(x.max() - x.min())

def display_image(output):
    output = output.reshape(output.shape[1] , output.shape[2] ,output.shape[3])
    output = output.permute(1,2,0)
    output = output.cpu()
    output = output.detach().numpy()
    output = norm(output)
    return output*255

def prepare_image(imag):
    imag = torch.tensor(imag)
    imag = norm(imag)
    imag = imag.permute(2,0,1)
    imag = imag.reshape(1,imag.shape[0] , imag.shape[1] ,imag.shape[2])
    return imag

input_path = 'test_input/'
depth_data_path = 'test_input_depth/'
data_name = os.listdir(input_path)
for img in data_name:
    if '.' in img:
        path = img
        print(path)
        img = cv.imread(f'{input_path}{img}')
        depth = cv.imread(f'{depth_data_path}/{path}')
        print(depth.shape)
        input_image = img[:,:,:]
        # input_image[:,:,0] = cv.equalizeHist(input_image[:,:,0])
        # input_image[:,:,1] = cv.equalizeHist(input_image[:,:,1])
        # input_image[:,:,2] = cv.equalizeHist(input_image[:,:,2])
        if '.png' in path:
            s = path.replace('.png' , '_output.png')
        elif '.jpeg' in path:
            s = path.replace('.jpeg' , '_output.png')
        elif '.bmp' in path:
            s = path.replace('.bmp' , '_output.png')
        else:
            s = path.replace('.jpg' , '_output.png')


        img = input_image
        # cv.imwrite(f'/home/cvg-ws05/msi_up/underwater_minor/REU_Sujay/data/cluster_restore/version_5/test_on_standard_dataset/test3/{s}' , img)

        input_image = prepare_image(input_image).to(device)
        depth = prepare_image(depth).to(device)

        # depth = prepare_image(depth).cpu()
        
        # inp_b = input_image[:,0,:,:].reshape(input_image.shape[0], 1 , input_image.shape[2] , input_image.shape[3]).to('cpu')
        # inp_g = input_image[:,1,:,:].reshape(input_image.shape[0] ,1 , input_image.shape[2] , input_image.shape[3]).to(device)
        # inp_r = input_image[:,2,:,:].reshape(input_image.shape[0] ,1 , input_image.shape[2] , input_image.shape[3]).to('cpu')


        # b = model_b(inp_b)
        # model_b = model_b.to('cpu')
        # inp_b = inp_b.to('cpu')
        # model_r = model_r
        # g = model_g(inp_g)  
        output_image = model_r(input_image , depth).to(device)

        # b = b.cpu()
        # g = g.cpu()
        # r = r.cpu()
        # img = torch.cat((b , g) , dim = 1)
        # img = torch.cat((img , r), dim = 1
        img = display_image(output_image)
        cv.imwrite(f'output/{s}' , img)
        print("testing complete")


    def hist_eq_them():
        data_name = os.listdir('/home/cvg-ws05/msi_up/underwater_minor/REU_Sujay/data/cluster_restore/version_5/test_on_standard_dataset/test3/')
        for i in data_name:
            if 'output' in i:
                input_image = cv.imread(f'/home/cvg-ws05/msi_up/underwater_minor/REU_Sujay/data/cluster_restore/version_5/test_on_standard_dataset/test3/{i}')
                input_image[:,:,0] = cv.equalizeHist(input_image[:,:,0])
                input_image[:,:,1] = cv.equalizeHist(input_image[:,:,1])
                input_image[:,:,2] = cv.equalizeHist(input_image[:,:,2])
                cv.imwrite(f'/home/cvg-ws05/msi_up/underwater_minor/REU_Sujay/data/cluster_restore/version_5/test_on_standard_dataset/test3/{i}' , input_image)
# hist_eq_them()
