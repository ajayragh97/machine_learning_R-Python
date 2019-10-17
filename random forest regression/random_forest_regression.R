#random forest regression


#import dataset
dataset = read.csv('Position_Salaries.csv')
#dataset = dataset[2:3]

#splitting dataset into training set and test set
#since the data is  small we wont be splitting it 

#feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])



#fitting  random forest regression model to dataset
#install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1],y = dataset$Salary, ntree = 10 )

#predicting a new result with the random forest regression model
y_pred = predict(regressor , data.frame(Level = 6.5))



#Visualising polynomial regression results(high resolution curve)
library(ggplot2)
#adding new extra levels
x_grid = seq(min(dataset$Level),max(dataset$Level),0.1)
ggplot()+
  geom_point(aes(x = dataset$Level , y = dataset$Salary), colour = 'red')+ 
  geom_line(aes(x= dataset$Level , y = predict(regressor, newdata = dataset)),colour = 'blue')+
  ggtitle('random forest regression model results')+
  xlab('xlabel')+
  ylab('ylabel')



