# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:16:45 2019

@author: 曾智源
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from dataloader import my_dataset
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from config import config
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class my_net(nn.Module):
    def __init__(self):
        super(my_net,self).__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),#inplace会覆盖原数据，省区反复申请，节省显存
                nn.BatchNorm2d(4),
                
                nn.Conv2d(in_channels=4,out_channels=8,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(8),
                
                nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(8),
                )
        self.fc = nn.Sequential(
                nn.Linear(in_features=8*100*100,out_features=500),
                nn.ReLU(inplace=True),
                
                nn.Linear(500,500),
                nn.ReLU(inplace=True),
                
                nn.Linear(500,5)
                )
    def forward_single(self,image):
        out = self.cnn(image)
        out = out.view(out.size()[0],-1)#拉成向量
        out = self.fc(out)
        return out
    def forward(self,input1,input2):
        output1 = self.forward_single(input1)
        output2 = self.forward_single(input2)
        return output1,output2

class Contrastive_loss(nn.Module):
    def __init__(self,margin=2.0):
        super(Contrastive_loss,self).__init__()
        self.margin = margin
    def forward(self,output1,output2,label):
        L2_distance = F.pairwise_distance(output1,output2,p=2)
        loss = torch.mean(label*torch.pow(L2_distance,2)+
                          (1-label)*torch.pow(torch.clamp(self.margin-L2_distance,min=0.0),2))
        return loss

import torch.optim as optim
def train(net,train_loader,train_epoch):
    criterion = Contrastive_loss()
    optimizer = optim.Adam(net.parameters(),lr=0.0005)
    running_loss = []
    counter = []
    iter_time = 0
    for epoch in range(train_epoch):
       
       for i,data in enumerate(train_loader,0):
           manhua,origin,label = data
           manhua,origin,label =manhua.cuda(),origin.cuda(),label.cuda()
           optimizer.zero_grad()
           out_m,out_o = net(manhua,origin)
           loss = criterion(out_m,out_o,label)
           loss.backward()
           optimizer.step()
           if i%100 == 0:
               print("epoch:{} loss:{}".format(epoch,loss.item()))
               iter_time+=100
               counter.append(iter_time)
               running_loss.append(loss.item())
    plt.plot(counter,running_loss)
    plt.show()
    torch.save(net,'net.pkl')
    return running_loss
    
if __name__ == "__main__":
    #import fire
    #fire.Fire()
    dataset = my_dataset(config.csv_file,transform=transforms.Compose([transforms.Resize((100,100)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))                                 
        ]))
    dataset.reprocess()
    dataloader = DataLoader(dataset,shuffle=True,num_workers=4,batch_size=64)
    net = my_net().to(device)
    #running_loss = train(my_net,dataloader,config.train_epoches)
    criterion = Contrastive_loss()
    optimizer = optim.Adam(net.parameters(),lr=0.0005)
    running_loss = []
    counter = []
    iter_time = 0
    for epoch in range(100):
       
       for i,data in enumerate(dataloader):
           manhua,origin,label = data
           manhua,origin,label =manhua.to(device),origin.to(device),label.to(device)
           #manhua,origin,label =manhua,origin,label
           optimizer.zero_grad()
           out_m,out_o = net(manhua,origin)
           loss = criterion(out_m,out_o,label)
           loss.backward()
           optimizer.step()
           if i%100 == 0:
               print("epoch:{} loss:{}".format(epoch,loss.item()))
               iter_time+=100
               counter.append(iter_time)
               running_loss.append(loss.item())
    plt.plot(counter,running_loss)
    plt.show()
    print(dataloader)
           
           
           
    

