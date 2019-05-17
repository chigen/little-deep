# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:47:51 2019

@author: 曾智源
"""
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.utils import shuffle
import random
import numpy as np
from config import config
import matplotlib.pyplot as plt
batch = 20
iid = 1000#1000个不同的人
csv_file = 'annotations.csv'


annotations = pd.read_csv('annotations.csv',header = None,names = ['image','type','id'])
image_name,image_type,image_id = annotations.iloc[:,0],annotations.iloc[:,1],annotations.iloc[:,2]
annotations = annotations.drop(index=[0])#去除第一行
#image_id = annotations.as_matrix(columns=['id'])
#feat_data = annotations.iloc[:,:][annotations[annotations.T.index[2]]=='3']

"""          
#image = Image.open('images/0/0/373038b76fdadc2761bc6b36e45da239.jpg')
#image.show()
#print(image_name[1])

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
        ])
image1 = image.resize((227,227))
#image1.show()
image2 = transform(image1)
#print(image2.shape)
    """
"""
数组组合方式
将数据每个人的漫画个数补全到和真人图一样多，复制成两组，一组打乱id，一组不动
"""
class my_dataset(Dataset):
    def __init__(self,csv_file,transform = None,train_mode=True):
        self.annotations = pd.read_csv(csv_file,header = None,names = ['image','type','id'])
        self.transform = transform
        self.train_mode = train_mode
        #self.image_name = self.annotations.iloc[:,0]
        #self.image_type = self.annotations.iloc[:,1]
        #self.image_id = self.annotations.iloc[:,2]
        self.labels = [0,1]
        self.label = None
        #self.reprocess()
    #预处理data
    def reprocess(self):
        manhua = pd.DataFrame(columns = ['image','type','id'])
        origin = pd.DataFrame(columns = ['image','type','id'])
        test_manhua = pd.DataFrame(columns = ['image','type','id'])
        test_origin = pd.DataFrame(columns = ['image','type','id'])
        for i in range(iid):#按每个id补齐图片
            i_image = annotations.iloc[:,:][annotations[annotations.T.index[2]]==str(i)]
            i_manhua = i_image.iloc[:,:][i_image[i_image.T.index[1]]=='0']
            i_origin = i_image.iloc[:,:][i_image[i_image.T.index[1]]=='1']
            len_m = len(i_manhua)
            len_o = len(i_origin)
            delta = len_o-len_m
            if delta>=0:
                i_manhua_delta = i_manhua.iloc[:delta,:]
                if delta<=len_m:            
                    manhua = manhua.append(i_manhua_delta,ignore_index = True)
                    manhua = manhua.append(i_manhua,ignore_index = True)
                else:
                    count = delta//len_m+1
                    res = delta%len_m
                    for i in range(count):
                        manhua = manhua.append(i_manhua,ignore_index = True)
                    manhua = manhua.append(i_manhua.iloc[:res,:],ignore_index = True)
                origin = origin.append(i_origin,ignore_index = True)
            if delta<0:
                delta = -delta
                i_origin_delta = i_origin.iloc[:delta,:]
                if delta<=len_o:
                    origin = origin.append(i_origin_delta,ignore_index=True)
                    origin = origin.append(i_origin,ignore_index=True)
                else:
                    count = delta//len_o+1
                    res = delta%len_o
                    for i in range(count):
                        origin = origin.append(i_origin_delta,ignore_index=True)
                    origin = origin.append(i_origin.iloc[:res,:],ignore_index=True)
                manhua = manhua.append(i_manhua,ignore_index = True)
        #manhua_shuffle = shuffle(manhua)
        #origin_shuffle = shuffle(origin)
        manhua_shuffle = manhua.sample(frac=1).reset_index(drop=True)
        origin_shuffle = origin.sample(frac=1).reset_index(drop=True)
        #添加label列
        manhua['label'] = '1'
        origin['label'] = '1'
        manhua_shuffle['label'] = '0'
        origin_shuffle['label'] = '0'
        data_len = 2*len(manhua)#所有数据个数
        test_index = random.sample(range(1,data_len),data_len//5)#随机选20%数据做test
        #拼接顺序的data和打乱的data作为train
        train_manhua = manhua.append(manhua_shuffle,ignore_index=True)
        train_origin = origin.append(origin_shuffle,ignore_index=True)
        #随机抽出data作为test
        test_manhua = test_manhua.append(train_manhua.iloc[test_index,:],ignore_index=True)
        test_origin = test_origin.append(train_origin.iloc[test_index,:],ignore_index=True)
        #删除被抽出的data
        train_manhua = train_manhua.drop(test_index)
        train_origin = train_origin.drop(test_index)
        #重设train的index
        train_manhua = train_manhua.reset_index(drop=True)
        train_origin = train_origin.reset_index(drop=True)
        self.train_origin = train_origin
        self.train_manhua = train_manhua
        self.test_manhua = test_manhua
        self.test_origin = test_origin
        print(train_manhua.shape)
        print(test_manhua.shape)
    def load_double_image(self,index):#默认加载train
        #image_pathes=[]
        self.label = None
        if self.train_mode:
            manhua = self.train_manhua.iloc[index]
            origin = self.train_origin.iloc[index]
            
        else:
            manhua = self.test_manhua.iloc[index]
            origin = self.test_origin.iloc[index]
        self.label = manhua.iloc[-1]
        manhua_path = 'images/'+str(manhua.id)+'/'+str(manhua.type)+'/'+str(manhua.image)
        #image_pathes.append(manhua_path)
        origin_path = 'images/'+str(origin.id)+'/'+str(origin.type)+'/'+str(origin.image)
        #image_pathes.append(origin_path)
        image_m = Image.open(manhua_path)
        image_o = Image.open(origin_path)
        image_m = image_m.convert("RGB")
        image_o = image_o.convert("RGB")
       
        return image_m,image_o
    def __getitem__(self,index):
        #manhua_same,origin_same,label_same = self.manhua[index],self.origin[index],self.labels[1]
        #manhua_diff,origin_diff,label_diff = self.manhua_shuffle[index],self.origin_shuffle[index],self.labels[0]
        manhua,origin = self.load_double_image(index)
        #manhua.show()
        #origin.show()
        if self.transform is not None:
            manhua = self.transform(manhua)
            origin = self.transform(origin)
        label = self.label
        label = torch.from_numpy(np.array([int(label)],dtype=np.float32))
        
        #print(label)
        return manhua,origin,label
    def __len__(self):
        if self.train_mode:
            return len(self.train_manhua)
        else:
            return len(self.test_manhua)

"""
dataset = my_dataset(csv_file,transform=transforms.Compose([transforms.Resize((100,100)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
                                                            
        ]))


for i,data in enumerate(dataset,0):
    manhua,origin,label = data
    print(manhua.shape)
"""
"""
dataloader = DataLoader(dataset,shuffle=False,num_workers=0,batch_size=24)
for i,data in enumerate(dataloader,0):
           manhua,origin,label = data
           print(manhua.shape)

dataset.reprocess()
dataset.__getitem__(24000)
dataset.__getitem__(2)
"""
