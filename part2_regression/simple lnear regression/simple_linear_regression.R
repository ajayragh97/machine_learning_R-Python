#simple linear regression model

#import dataset
dataset = read.csv('Salary_Data.csv')


#splitting dataset into training set and test set
#install.packages('caTools') 
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary,SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

#fitting simple linear regression model to training set
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

#predicting test set results
y_pred = predict(regressor, newdata = test_set) 

#visualising training set results
# uncooment the next line if ggplot is not installed 
#install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y= training_set$Salary),
             colour = 'red') +
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs experience(training set)')+
  xlab('years of experience')+
  ylab('salary')

#visualising testg set results
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y= test_set$Salary),
             colour = 'red') +
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs experience(test set)')+
  xlab('years of experience')+
  ylab('salary')
