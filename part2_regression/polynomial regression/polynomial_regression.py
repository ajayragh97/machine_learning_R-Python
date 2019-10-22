# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 15:16:10 2019

@author: Ajay Ragh
"""

#polynomial regression
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#import dataset

dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values

#fitting dataset to linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#fitting dataset to polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

#visualising linear regression results
plt.scatter(x, y,color = 'red')
plt.plot(x, lin_reg.predict(x),color = 'blue')
plt.title('linear regression plot')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#visualising polynomial regression results

#controlling resolution of the curve
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))

plt.scatter(x, y,color = 'red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color = 'blue')
plt.title('polynomial regression plot')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#predicting results with linear regression model
lin_reg.predict(6.5)

#predicting results with polynomial regression model
lin_reg_2.predict(poly_reg.fit_transform(6.5))