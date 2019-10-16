#decision tree rgeression
#import dataset
dataset = read.csv('Position_Salaries.csv')
#dataset = dataset[2:3]

#splitting dataset into training set and test set
#since the data is  small we wont be splitting it 

#feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])



#fitting  regression model to dataset
#create regressor hhere
#uncomment next install line if package is not preinstalled
#install.packages('rpart')
library(rpart)
regressor = rpart(formula = Salary ~ Level , data = dataset, control = rpart.control(minsplit = 1))

#predicting a new result with the  regression model
y_pred = predict(regressor , data.frame(Level = 6.5))


#Visualising polynomial regression results
library(ggplot2)
#adding new extra levels
x_grid = seq(min(dataset$Level),max(dataset$Level),0.01)
ggplot()+
  geom_point(aes(x = dataset$Level , y = dataset$Salary), colour = 'red')+ 
  geom_line(aes(x= x_grid , y = predict(regressor, newdata = data.frame(Level = x_grid))),colour = 'blue')+
  ggtitle('decision tree regression model results')+
  xlab('xlabel')+
  ylab('ylabel')


