# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 18:29:25 2019

@author: Ajay Ragh
"""
#Data Preprocessing

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
dataset=pd.read_csv('Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 3].values

#Splitting dataset into training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test) """
