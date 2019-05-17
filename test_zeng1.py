# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 08:57:04 2019

@author: 11732
"""

import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
import torch.functional as F
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn import metrics
net = torch.load('net.pkl')
csv_file = 'list.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class my_net(nn.Module):
    def __init__(self):
        super(my_net,self).__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=4,kernel_size=3,padding=1),
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

class my_test_dataset(Dataset):
    def __init__(self,csv_file,transform = None):
        super(my_test_dataset,self).__init__()
        self.transform = transform
        self.list = pd.read_csv(csv_file,header =None,names=['gruop_id','image1','image2'])
        print(self.list)
    def load(self,index):
        test_i = self.list.iloc[index]
        image1_path = 'test_images/'+str(test_i.image1)
        image2_path = 'test_images/'+str(test_i.image2)
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)
        image1 = image1.convert("RGB")
        image2 = image2.convert("RGB")
        return image1,image2
    def __getitem__(self,index):
        image1,image2 = self.load(index)
        #image1.show()
        #image2.show()
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        g_id = self.list.iloc[index,2]
        g_id = torch.from_numpy(np.array([int(g_id)],dtype=np.float32))
        #print(g_id)
        return image1,image2,g_id
    def __len__(self):
        return len(self.list.image1)

net = torch.load('net.pkl')

#net = torch.load('net.pkl')
"""
if __name__ == "__main__":
    test_dataset = my_test_dataset(csv_file,transform=transforms.Compose([transforms.Scale((100,100)),transforms.ToTensor()]))
    
    test_dataloader = DataLoader(test_dataset,batch_size=1,num_workers=0,shuffle=False)
    net = net()
    pred = []
    group_id = []
    for i,data in enumerate(test_dataloader):
        image1,image2,g_id=data
        image1,image2,g_id=image1.to(device),image2.to(device),g_id.to(device)
        output1,output2=net(image1,image2)
        L2_distance = F.pairwise_distance(output1,output2,p=2)
        pred.append(L2_distance)
        group_id.append(g_id)
    df = pd.DataFrame({'group_id':group_id,'confidence':pred})
    df.to_csv("predictions.csv",index=False,sep=',')
#test_dataset.__getitem__(3)
    """

    
    
