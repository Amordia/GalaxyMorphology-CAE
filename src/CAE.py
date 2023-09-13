#! -*- coding: utf-8 -*-
##特征提取
import pylab as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.table import Table
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import *
import torch.nn.functional as F

import os

batch_size_test = 128
batch_size_train = 32
img_size = 64

test_no = 4.1

num_epoch = 50
log_interval = 10

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


trainData = ImgDataset('C:\\Users\\23650\\Desktop\\research\\NAO\\datasets\\trainSet.hdf5')
trainLoader = torch.utils.data.DataLoader(
    dataset=trainData, batch_size=batch_size_train, shuffle=True
)

trainSetLen = trainData.__len__()

testData = ImgDataset('C:\\Users\\23650\\Desktop\\research\\NAO\\datasets\\testSet.hdf5')
testLoader = torch.utils.data.DataLoader(
    dataset=testData, batch_size=batch_size_test, shuffle=False
)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 9, stride=1, padding=4),  # in_channels=3 out_channels=32 kernel_size=9
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), #32 # kernel_size=2
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

def train(epoch):
    
    print('start train')
    for batch_idx, (img) in enumerate(trainLoader):
             
        img = img.cuda()
               
        encode,decode = AE(img)

        loss = criterion_MSE(decode,img)
        
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()
            
        #scheduler.step(epoch + batch_idx / iters)
            
        if (batch_idx + 1) % log_interval == 0:          
            
            print('Epoch[{}/{}],loss:{:.6f}'.format(
                epoch, num_epoch,loss.data.item()
            ))
            train_losses.append(loss.data.item())
            train_counter.append(
                #(batch_idx*batch_size_train)/len(train_loader.dataset) + (epoch))  
                (batch_idx*batch_size_train)/trainSetLen + (epoch))         
                                    
    print('end train') 

def test(epoch):
    result_fig = []
    with torch.no_grad():
        test_loss_MSE = 0
        print('start test')
        for batch_idx, (img) in enumerate(testLoader):
            img = img.cuda()
               
            encode,decode = AE(img)
            loss_MSE = criterion_MSE(decode,img)
            temp_MSE = loss_MSE.data.item()
            test_loss_MSE += batch_size_test * temp_MSE   

            if epoch % 5 == 0:    
                result = decode.cpu().detach()[0].numpy()
                result_fig.append(result)
                
        test_loss_MSE /= len(testLoader.dataset)
        print('MSE_loss:{:.6f}'.format(test_loss_MSE))
        test_losses_MSE.append(test_loss_MSE)
        test_counter.append(epoch)
        if epoch % 5 == 0:            
            t = Table(rows=result_fig,names=('r','g','b'))
            t.write('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\result-{}.hdf5'.format(test_no,epoch))
        print('end test')
    
AE = autoencoder().cuda()

'''
def G_loss_function(x,y):   
    #return torch.mean(torch.pow((x - y) /(x+y),2) * (x + y)) * torch.mean(x + y)
    #return 100 * torch.mean(torch.pow((x-y),2) * (x + y))
    return torch.mean(torch.pow((x - y),2) * torch.exp(x - y))
'''
optimizer = torch.optim.Adam(AE.parameters(), lr=0.0005)
criterion_MSE = nn.MSELoss(reduction = 'mean')

train_losses = []
train_counter = []
test_losses_MSE = []
test_counter = []

for epoch in range(num_epoch):   
    train(epoch)
    if (epoch) % 5==0: 
        torch.save(AE.state_dict(), 'C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\CAE-{}'.format(test_no,epoch) +  '.pth')    
    test(epoch)    
    
    mat = np.array(train_counter)
    df = pd.DataFrame(mat)
    df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\train_counter.csv'.format(test_no),index=False,sep=',')   
    mat = np.array(train_losses)
    df = pd.DataFrame(mat)
    df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\train_losses.csv'.format(test_no),index=False,sep=',')   
    mat = np.array(test_counter)
    df = pd.DataFrame(mat)
    df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\test_counter.csv'.format(test_no),index=False,sep=',')   
    mat = np.array(test_losses_MSE)
    df = pd.DataFrame(mat)
    df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\test_losses_MSE.csv'.format(test_no),index=False,sep=',') 