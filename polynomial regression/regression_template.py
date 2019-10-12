# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 16:41:27 2019

@author: Ajay Ragh
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#import dataset

dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values

#Splitting dataset into training set and test set
"""from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3,random_state=0)"""

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test) """

#fitting dataset to  regression model


#predicting results with polynomial regression model
y_pred = regressor.predict(x)

#visualising the regression model results
plt.scatter(x, y,color = 'red')
plt.plot(x,regressor.predict(x),color = 'blue')
plt.title('regression model plot')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#visualising the regression model results(for higher resolution curve)
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y,color = 'red')
plt.plot(x_grid,regressor.predict(x_grid),color = 'blue')
plt.title('regression model plot')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

