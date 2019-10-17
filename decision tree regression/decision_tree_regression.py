# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 20:43:24 2019

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
sc_y = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test) """

#fitting dataset to  regression model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x,y)


#predicting results with decision tree regression model
y_pred = regressor.predict(6.5)

#visualising the regression model results
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y,color = 'red')
plt.plot(x_grid,regressor.predict(x_grid),color = 'blue')
plt.title('regression model plot')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()