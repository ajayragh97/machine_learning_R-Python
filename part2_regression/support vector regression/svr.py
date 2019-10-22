# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:49:35 2019

@author: Ajay Ragh
"""

#SVR
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#import dataset

dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2:3].values

#Splitting dataset into training set and test set
"""from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3,random_state=0)"""

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

#fitting dataset to support vector regression model
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

#predicting results with polynomial regression model
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

#visualising the regression model results
plt.scatter(x, y,color = 'red')
plt.plot(x,regressor.predict(x),color = 'blue')
plt.title('SVR model plot')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()