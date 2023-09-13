#! -*- coding: utf-8 -*-
##特征提取
import pylab as pl
import numpy as np
import pandas as pd

from astropy.table import Table
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import *
import torch.nn.functional as F

import os

batch_size_test = 100
img_size = 64

test_no = 4

torch.cuda.set_device(0)
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(2)

class ImgDataset(Dataset):
    def __init__(self,hdf5_file):
        data = Table.read(hdf5_file,path="/data")
        
        self.imgs = []
        
        print("------开始读取数据------")   
        for i in range(len(data)):
    
            #-------------------------------------------------
            # load in fimgs
            #
            img_r = data['r'][i]
            img_g = data['g'][i]
            img_b = data['b'][i]

            #-------------------------------------------------
            # rescale and clip fimgs according to the above 
            # scale factor and thresholds
            #          
            
            img_r_rscl = img_r + 0.5
            img_g_rscl = img_g + 0.5
            img_b_rscl = img_b + 0.5
            
            #-------------------------------------------------
            # determine scale factors and thresholds
            #
                        
            img = np.zeros([3,64,64])
            img[0,:,:] = img_r_rscl
            img[1,:,:] = img_g_rscl
            img[2,:,:] = img_b_rscl
            
            self.imgs.append(torch.Tensor(img))
        
        print("------读取文件结束------")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        return img


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 9, stride=1, padding=4),  
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  #32
            nn.Conv2d(32, 32, 7, stride=1, padding=3),  
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  #16
            nn.Conv2d(32, 16, 5, stride=1, padding=2),  
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  #8
            nn.Conv2d(16, 8, 3, stride=1, padding=1),  
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  #4          
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2,padding=0),  #9
            nn.ReLU(True),  
            nn.ConvTranspose2d(16, 32, 5, stride=2,padding=2),  #17
            nn.ReLU(True), 
            nn.ConvTranspose2d(32, 32, 7, stride=2,padding=3),  #33
            nn.ReLU(True), 
            nn.ConvTranspose2d(32, 3, 9, stride=2, padding=4), #65
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1), 
            nn.Tanh()
        )
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

def Test(loader,name):
    
    print('start test')   
    result_array = []
    result_fig = []
    with torch.no_grad():
        test_loss_MSE = 0
        print('start test')
        for batch_idx, (img) in enumerate(loader):
            img = img.cuda()
            results,decode = AE(img)
                
            for i in range(batch_size_test):
                result = results.view(-1,128).cpu().detach()[i].numpy()
                result_array.append(result)
                result_img = decode.cpu().detach()[i].numpy()
                result_fig.append(result_img)
        
        mat = np.array(result_array)
        t = Table(rows=result_fig,names=('r','g','b'))
        t.write('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\result_{}.hdf5'.format(test_no,name),path='\\data')
        df = pd.DataFrame(mat)
        df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\feature_{}.csv'.format(test_no,name),index=False,sep=',')   
                                    
    print('end test') 
    
AE = autoencoder().cuda()
model_path = 'C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\scripts2\\CAE.pth'
AE.load_state_dict(torch.load(model_path,map_location = 'cuda:0'))

criterion_MSE = nn.MSELoss(reduction = 'mean')

valData = ImgDataset('C:\\Users\\23650\\Desktop\\research\\NAO\\datasets\\valSet.hdf5')
valLoader = torch.utils.data.DataLoader(
    dataset=valData, batch_size=batch_size_test, shuffle=False
)

testData = ImgDataset('C:\\Users\\23650\\Desktop\\research\\NAO\\datasets\\testSet.hdf5')
testLoader = torch.utils.data.DataLoader(
    dataset=testData, batch_size=batch_size_test, shuffle=False
)

Test(valLoader,'valSet')
Test(testLoader,'testSet')