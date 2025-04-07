from ast import Gt
# from cv2 import detail_GainCompensator
from matplotlib import axis
import torch.nn as nn
import torch.nn.functional as F
import torch
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
from pytorch_msssim import ssim
import restoreNet

def tensor_plt(tens):
  tens = tens.permute(1,2,0)
  nparray = np.array(tens)
  cv.imshow('image' , nparray)
  cv.waitKey(0)  
  cv.destroyAllWindows()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class Dataset(torch.utils.data.Dataset):
  #Characterizes a dataset for PyTorch
    def __init__(self, csv_file, root_directory_ground_truth,root_directory_input , root_directory_depth):
        #Initialization
        self.annotations = pd.read_csv(csv_file , header = None)
        self.root_directory_input = root_directory_input
        self.root_directory_ground_truth = root_directory_ground_truth
        self.root_directory_depth = root_directory_depth

    def __len__(self):
        #Denotes the total number of samples'
        return len(self.annotations)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.root_directory_input, self.annotations.iloc[index,0])
        groundtruth_img_path = os.path.join(self.root_directory_ground_truth,self.annotations.iloc[index,0])
        depth_img_path = os.path.join(self.root_directory_depth,self.annotations.iloc[index,0])
        
        input_image = cv.imread(input_img_path)
        label_image = cv.imread(groundtruth_img_path).astype(np.float32)
        depth_image = cv.imread(depth_img_path).astype(np.float32)

        input_image = input_image.astype(np.float32)

        input_image = torch.from_numpy(input_image)
        label_image = torch.from_numpy(label_image)
        depth_image = torch.from_numpy(depth_image)


        input_image = input_image.permute(2,0,1)
        label_image = label_image.permute(2,0,1)
        depth_image = depth_image.permute(2,0,1)

        # print( self.annotations.iloc[index,0] , 'remove this' , index)

        input_image = input_image/255.0
        label_image = label_image/255.0
        depth_image = depth_image/255.0

        return (input_image , label_image  , depth_image)

dataset = Dataset(csv_file = '/workspace/codebase/synthetic_restore/version_3/files_for_training/train_data.csv',
    root_directory_input = '/workspace/data/uieb_jerlov_synthetic_data/UIEB_Synthetic_input/',
    root_directory_ground_truth = '/workspace/data/uieb_jerlov_synthetic_data/UIEB_Synthetic_label/' , 
    root_directory_depth = "/workspace/data/uieb_jerlov_synthetic_data/UIEB_Synthetic_depth/")

print(len(dataset) , 'length of dataset')
inp ,lab , depthh= dataset[30]
print(inp.shape , lab.shape , 'shape of input and label')
batch_size = 1
data_loader = torch.utils.data.DataLoader(dataset,batch_size = batch_size,shuffle = False)
data_iter = iter(data_loader)
print(len(data_loader))
inp  , lab  , deppp =  data_iter.next()
print(inp.shape , lab.shape , deppp.shape)
print(inp[0].min() ,inp[0].max(), lab[0].min(),lab[0].max())

print('Note, This is training of synthetic restore version_3 , class_all')

model_r = restoreNet.generator()
model_r = model_r.to(device)

epoch_start = 0
load_model = 0
if load_model:
    model_r = torch.load('/workspace/codebase/synthetic_restore/version_3/latest_2_r.pth').to(device)
    epoch_start = 0
    print('---models loaded--- , epoch start - 0')
else:
    print('model started training from beginning')

layers_s = [0,1,2]
criterion_r = nn.MSELoss()
optimizer_r = torch.optim.Adam(model_r.parameters(),lr = 0.0005)

def calculate_ssim_loss(label , op):
    op = op.reshape( op.shape[0] , op.shape[1] , op.shape[2] , op.shape[3]  )
    label = label.reshape( label.shape[0] , label.shape[1] , label.shape[2],label.shape[3])
    loss = abs(ssim(op, label, data_range=1, size_average=False))
    return 1-loss

def test_now(epoch , c , data_set = dataset):
    def norm(x):
        return (x-x.min())/(x.max() - x.min())

    def display_image(output):
        output = output.reshape(output.shape[1] , output.shape[2] ,output.shape[3])
        output = output.permute(1,2,0)
        output = output.cpu()
        output = output.detach().numpy()
        return output*255

    def prepare_image(imag):
        imag = torch.tensor(imag)
        imag = norm(imag)
        imag = imag.permute(2,0,1)
        imag = imag.reshape(1,imag.shape[0] , imag.shape[1] ,imag.shape[2])
        return imag

    val = torch.randint(0 ,len(data_set) - 1 , (1,))
    val = int(val)
    data_list = pd.read_csv('/workspace/codebase/synthetic_restore/version_3/files_for_training/train_data.csv' , header = None)
    img = cv.imread(f'/workspace/data/uieb_jerlov_synthetic_data/UIEB_Synthetic_input/{data_list.iloc[val , 0]}')
    gt = cv.imread(f'/workspace/data/uieb_jerlov_synthetic_data/UIEB_Synthetic_label/{data_list.iloc[val , 0]}')

    ip_image,_ , dep  = data_set[val]

    cv.imwrite(f'/workspace/codebase/synthetic_restore/version_3/results/{epoch}_{c}_label.png',gt)
    cv.imwrite(f'/workspace/codebase/synthetic_restore/version_3/results/{epoch}_{c}_input.png',img)

    ip_image = ip_image.reshape(1 , ip_image.shape[0] , ip_image.shape[1], ip_image.shape[2])
    ip_image = ip_image.to(device)
    dep = dep.reshape(1 , dep.shape[0] , dep.shape[1], dep.shape[2])
    dep = dep.to(device)

    img = model_r(ip_image , dep)
    img = display_image(img)

    cv.imwrite(f'/workspace/codebase/synthetic_restore/version_3/results/{epoch}_{c}_output.png',img)
    print('----------Image written----------')

test_now(0 , 1)
best_loss = 1000
no_epoch = 1000000

# training loop 
for epoch in range(no_epoch):
    avg_loss = 0
    c = 0
    avg_r = 0
    avg_g = 0
    avg_b = 0
    for i , data in enumerate(data_loader):
        inp_img , lab_img  , dep_img  = data
        inp_img ,lab_img , dep_img =  inp_img.to(device) , lab_img.to(device) , dep_img.to(device)
        r = model_r(inp_img , dep_img)
        loss_r = criterion_r(lab_img , r)

        loss_r.backward()
        optimizer_r.step()
        optimizer_r.zero_grad()

        avg_r = avg_r + loss_r.item()
        c = c + 1
        if c % 100 == 0:
            print(f"{c} iterations over-----")
            print(avg_r/c , "avg" )
        if c % 50 == 0:
            if epoch < 5:
                print(f"{c} iterations over-----") 
                test_now(epoch + epoch_start + 1, c)
            else:
                if c % 500 == 0:
                    print(f"{c} iterations over-----") 
                    test_now(epoch + epoch_start +1, c)

    if (epoch) % 2 == 0:
        torch.save(model_r , f"/workspace/codebase/synthetic_restore/version_3/models/{epoch + epoch_start}_class_2_r.pth")
        print('---------model saved----------')
    torch.save(model_r , f"/workspace/codebase/synthetic_restore/version_3/latest_2_r.pth")
    test_now(epoch + epoch_start+1,c)