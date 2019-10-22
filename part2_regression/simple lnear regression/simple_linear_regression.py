# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 19:21:58 2019

@author: Ajay Ragh
"""

#Simple linear regression
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values

#Splitting dataset into training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3,random_state=0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test) """

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting test set results
y_pred = regressor.predict(x_test)

#visualising the training set results
plt.scatter(x_train,y_train, color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('salary vs experience(training set)')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()

#visualising the test set results
plt.scatter(x_test,y_test, color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('salary vs experience(test set)')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()