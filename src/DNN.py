#! -*- coding: utf-8 -*-
##特征提取后分类
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

batch_size_test = 200
batch_size_train = 100

test_no = 4

num_epoch = 50
log_interval = 1

torch.cuda.set_device(0)
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(2)

class FeatureDataset(Dataset):
    def __init__(self,feature_file,label_file):
        self.features = np.array(pd.read_csv(feature_file))
        self.labels = np.array(pd.read_csv(label_file))      
        print("------读取文件结束------")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        feature = torch.Tensor(self.features[idx])
        label = torch.LongTensor(self.labels[idx])
        return feature,label

feature_dir = 'C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\'
label_dir = 'C:\\Users\\23650\\Desktop\\research\\NAO\\datasets\\'
trainData = FeatureDataset(feature_dir + '{}\\feature_valSet.csv'.format(test_no),label_dir + 'valSet.csv')
trainLoader = torch.utils.data.DataLoader(
    dataset=trainData, batch_size=batch_size_train, shuffle=True
)

testData = FeatureDataset(feature_dir + '{}\\feature_testSet.csv'.format(test_no),label_dir + 'testSet.csv')
testLoader = torch.utils.data.DataLoader(
    dataset=testData, batch_size=batch_size_test, shuffle=False
)

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(128,128),
            nn.PReLU(), 
            nn.Linear(128,128),
            nn.PReLU(),
            nn.Linear(128,64),
            nn.PReLU(),
            nn.Linear(64,32),
            nn.PReLU(),
            nn.Linear(32,4),
            nn.LogSoftmax()      
        )
    def forward(self, x):
        x = self.dense(x)
        return x

def train(epoch):
    
    print('start train')
    for batch_idx, (feature,label) in enumerate(trainLoader):
            
        feature = feature.cuda()
        label = label.cuda().view(-1)
               
        result = D(feature)
        loss = criterion(result,label)
        
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
                (batch_idx*batch_size_train)/len(trainLoader.dataset) + (epoch))        
                                    
    print('end train') 

def test(epoch):
    with torch.no_grad():
        test_loss_MSE = 0
        print('start test')
        for batch_idx, (feature,label) in enumerate(testLoader):
            feature = feature.cuda()
            label = label.cuda().view(-1) 
            
            result = D(feature)
            loss_MSE = criterion(result,label)
            temp_MSE = loss_MSE.data.item()
            test_loss_MSE += batch_size_test * temp_MSE   
                
        test_loss_MSE /= len(testLoader.dataset)
        print('MSE_loss:{:.6f}'.format(test_loss_MSE))
        test_losses_MSE.append(test_loss_MSE)
        test_counter.append(epoch)

        print('end test')
    
D = DNN().cuda()

optimizer = torch.optim.Adam(D.parameters(), lr=0.0005)
criterion = nn.NLLLoss()

train_losses = []
train_counter = []
test_losses_MSE = []
test_counter = []

for epoch in range(num_epoch):   
    train(epoch)
    if (epoch) % 5==0: 
        torch.save(D.state_dict(), 'C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\DNN-{}'.format(test_no,epoch) +  '.pth')    
    test(epoch)    
    
    mat = np.array(train_counter)
    df = pd.DataFrame(mat)
    df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\DNNtrain_counter.csv'.format(test_no),index=False,sep=',')   
    mat = np.array(train_losses)
    df = pd.DataFrame(mat)
    df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\DNNtrain_losses.csv'.format(test_no),index=False,sep=',')   
    mat = np.array(test_counter)
    df = pd.DataFrame(mat)
    df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\DNNtest_counter.csv'.format(test_no),index=False,sep=',')   
    mat = np.array(test_losses_MSE)
    df = pd.DataFrame(mat)
    df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\DNNtest_losses_MSE.csv'.format(test_no),index=False,sep=',')    
