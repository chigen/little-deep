# -*- coding: utf-8 -*-
"""
Created on Fri May 17 23:45:02 2019

@author: 曾智源
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

train = pd.read_csv('train_data.csv')
#test = pd.read_csv('test_a.csv')
#print(train.sample(5))
X = train.iloc[:,:-1].values
Y = train.iloc[:,-1].values
drop_feat = ['ID','communityName','city','region','plate','tradeTime','houseType']
#onehot_feat = ['rentType','houseFloor','houseToward','houseDecoration']
onehot_feat = [4,5,7,8]
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
house_type = train['houseType'].str.split('室',expand=True)
#encoding categorial data
def pre_cate(df):
    labelencoder_X = LabelEncoder()
    house_type = df['houseType'].str.split('室',expand=True)
    bedroom,livingroom,restromm = house_type.pop(0),house_type.pop(1),house_type.pop(2)
    df.insert(0,'bedroom',bedroom)
    df.insert(1,'livingroom',livingroom)
    df.insert(2,'restroom',restromm)
    #df['rentType'] = df['rentType'].map({'未知方式':'unknown','整租':'entire','合租':'joint'})
    #df['houseFloor'] = df['houseFloor']
    df = df.drop(drop_feat,axis =1)
    X = df.iloc[:,:-1].values
    Y = df.iloc[:,-1].values
    imputer = Imputer(missing_values = "暂无信息", strategy = "mean", axis = 0)
    imputer = imputer.fit(X[ : , 1:3])
    X[ : , 1:3] = imputer.transform(X[ : , 9])
    X[:,onehot_feat] = labelencoder_X.fit_transform(X[:,onehot_feat])
    #creating dummy variable
    onehotencoder = OneHotEncoder(categorical_features=[0])
    X = onehotencoder.fit_transform(X).toarray()
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)
    return X_train,X_test,Y_train,Y_test

#X_train,X_test,Y_train,Y_test = pre_cate(train)