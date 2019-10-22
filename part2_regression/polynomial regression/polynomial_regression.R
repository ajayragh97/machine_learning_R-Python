#polynomial regression
#import dataset
dataset = read.csv('Position_salaries.csv')
dataset = dataset[2:3]

#splitting dataset into training set and test set
#since the data is  small we wont be splitting it 

#feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

#fitting linear model to dataset(for comparison with polynomial regression model)
lin_reg = lm(formula = Salary ~ .,data = dataset)

#fitting polynomial regression model to dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ ., data = dataset)

#visualising linear regression model results
library(ggplot2)
ggplot()+
  geom_point(aes(x = dataset$Level , y = dataset$Salary),colour = 'red')+
  geom_line(aes(x = dataset$Level , y = predict(lin_reg, newdata = dataset)),colour = 'blue')+
  ggtitle('linear regression results')+
  xlab('Levels')+
  ylab('salary')

#Visualising polynomial regression results
ggplot()+
  geom_point(aes(x = dataset$Level , y = dataset$Salary), colour = 'red')+ 
  geom_line(aes(x= dataset$Level , y = predict(poly_reg, newdata = dataset)),colour = 'blue')+
  ggtitle('polynomial regression model results')+
  xlab('levels')+
  ylab('Salary')

#predicting a new result with the linear model
y_pred = predict(lin_reg, data.frame(Level = 6.5))

#predicting a new result with the polynomial regression model
y_pred = predict(poly_reg , data.frame(Level = 6.5, Level2 = 6.5^2, Level3 = 6.5^3, Level4 = 6.5^4))
