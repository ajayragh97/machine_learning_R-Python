#SVR
#import dataset
dataset = read.csv('Position_Salaries.csv')
#dataset = dataset[2:3]

#splitting dataset into training set and test set
#since the data is  small we wont be splitting it 

#feature scaling
#dataset[, 2:3] = scale(dataset[, 2:3])


#fitting  regression model to dataset
#uncomment the install line if the package is not pre-installed
#install.packages('e1071')
library(e1071)
regressor = svm(formula = Salary ~ Level,data = dataset,type = 'eps-regression')

#predicting a new result with the  regression model
y_pred = predict(regressor , data.frame(Level = 6.5))


#Visualising polynomial regression results
ggplot()+
  geom_point(aes(x = dataset$Level , y = dataset$Salary), colour = 'red')+ 
  geom_line(aes(x= dataset$Level , y = predict(regressor, newdata = dataset)),colour = 'blue')+
  ggtitle('SVR model results')+
  xlab('xlabel')+
  ylab('ylabel')
