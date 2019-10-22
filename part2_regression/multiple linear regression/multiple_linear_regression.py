# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:15:09 2019

@author: Ajay Ragh
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
dataset=pd.read_csv('50_Startups.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 4].values

#Encoding categorical datas(labels) into numbers for calculations
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
x = onehotencoder.fit_transform(x).toarray()

#Avoiding the dummy variable trap
x = x[:, 1:]

#Splitting dataset into training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)

#fitting multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting test set results
y_pred = regressor.predict(x_test)

#building optimal model using backward elimination
#we fix significance level(SL)=0.05
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)
#fitting full model with all possible predictors
x_opt = x[:, [0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y,exog = x_opt).fit()
regressor_ols.summary()

#removing predictor with P > (SL=0.05)
#round_1
x_opt = x[:, [0,1,3,4,5]]
regressor_ols = sm.OLS(endog = y,exog = x_opt).fit()
regressor_ols.summary()

#round_2
x_opt = x[:, [0,3,4,5]]
regressor_ols = sm.OLS(endog = y,exog = x_opt).fit()
regressor_ols.summary()

#round_3
x_opt = x[:, [0,3,5]]
regressor_ols = sm.OLS(endog = y,exog = x_opt).fit()
regressor_ols.summary()

#round_4
x_opt = x[:, [0,3]]
regressor_ols = sm.OLS(endog = y,exog = x_opt).fit()
regressor_ols.summary()