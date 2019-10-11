#multiple linear regression

#import dataset
dataset = read.csv('50_Startups.csv')
#dataset = dataset[, 2:3]

#encoding the categorical variable
dataset$State = factor(dataset$State, levels = c('New York','California','Florida'),labels = c(1,2,3))

#splitting dataset into training set and test set
#install.packages('caTools') 
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit,SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#fit multiple linear regression to training set
regressor = lm(formula = Profit ~ ., data = training_set)

#checking for the significance levels of the variables
summary(regressor)

#predicting test set results
y_pred = predict(regressor,newdata = test_set)

#building optimal model using backward elimination

#round1
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = dataset)
#checking significance levels and removing independent variables with p value above 0.05
summary(regressor)

#round2
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, data = dataset)
summary(regressor)

#round3
regressor = lm(formula = Profit ~ R.D.Spend +  Marketing.Spend, data = dataset)
summary(regressor)

#round4
regressor = lm(formula = Profit ~ R.D.Spend, data = dataset)
summary(regressor)
