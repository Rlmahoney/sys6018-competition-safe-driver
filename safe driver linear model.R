#Sys 6018
#Rob Mahoney: rlm4bj
#safe driver linear model

library(MASS)
library(ggplot2)
library(ISLR)
library(randomForest)
library(tree)
library(gbm)
library(glmnet)
library(RSQLite)
library(tidyverse)
library(DAAG)
library(boot)

#########################################################################################################
#                                                 Cleaning
#########################################################################################################

train = read_csv('train.csv')
test = read_csv('test.csv')

#--------------------------------------------------missings----------------------------------------------

#get proportions of missing values within variables
missing_prop = apply(train, 2, function(x) sum(x == -1)/nrow(train))
#only care about variables with any missing at all
missing_prop = missing_prop[missing_prop > 0]


#see the percent of each variable that's missing
for(i in 1:length(missing_prop)) {
  print(sprintf('%s:..................%f%% missing', names(missing_prop)[i], missing_prop[i]*100))
  flush.console
}

#drop variables with large numbers of missings
train = train[, -which(names(train) == 'ps_car_03_cat')]
train = train[, -which(names(train) == 'ps_car_05_cat')]

test = test[, -which(names(test) == 'ps_car_03_cat')]
test = test[, -which(names(test) == 'ps_car_05_cat')]

#continuous variables should be imputed by mean, but the -1 shifts the mean so it needs to be replaced
train$ps_reg_03[train$ps_reg_03 == -1] = NA
train$ps_car_12[train$ps_car_12 == -1] = NA
train$ps_car_14[train$ps_car_14 == -1] = NA

test$ps_reg_03[test$ps_reg_03 == -1] = NA
test$ps_car_12[test$ps_car_12 == -1] = NA
test$ps_car_14[test$ps_car_14 == -1] = NA

#imputing mean for continuous and ordinal variables
train$ps_reg_03[is.na(train$ps_reg_03)] = mean(train$ps_reg_03, na.rm = TRUE)
train$ps_car_12[is.na(train$ps_car_12)] = mean(train$ps_car_12, na.rm = TRUE)
train$ps_car_14[is.na(train$ps_car_14)] = mean(train$ps_car_14, na.rm = TRUE)

test$ps_reg_03[is.na(test$ps_reg_03)] = mean(train$ps_reg_03, na.rm = TRUE)
test$ps_car_12[is.na(test$ps_car_12)] = mean(train$ps_car_12, na.rm = TRUE)
test$ps_car_14[is.na(test$ps_car_14)] = mean(train$ps_car_14, na.rm = TRUE)

#imputing mode for factors/ordinal

#mode function
mode = function(x) {
  unique_x = unique(x)
  return(unique_x[which.max(tabulate(match(x, unique_x)))])
}

#re-calculate variables with missing values
missing_prop = apply(train, 2, function(x) sum(x == -1)/nrow(train))
missing_prop = missing_prop[missing_prop > 0]

#see the percent of each variable that's missing
for(i in 1:length(missing_prop)) {
  print(sprintf('%s:..................%f%% missing', names(missing_prop)[i], missing_prop[i]*100))
  flush.console
}

#replace with mode
for(i in 1:length(missing_prop)) {
  train[train[, names(missing_prop)[i]] == -1, names(missing_prop)[i]] = mode(unlist(train[,names(missing_prop)[i]]))
  test[test[, names(missing_prop)[i]] == -1, names(missing_prop)[i]] = mode(unlist(train[,names(missing_prop)[i]]))
  
}

#------------------------------------------------data types--------------------------------------------------

#making sure there are no variables in only one set
setdiff(names(train), names(test))

#get all variables labelled as either binary or categorical
factor_cols = names(train)[grep('bin|cat', names(train))]

#add target variable for training data
factor_cols_train = c('target', factor_cols)

#convert to factor
train[factor_cols_train] = lapply(train[factor_cols_train], factor)
test[factor_cols] = lapply(test[factor_cols], factor)

#check cardinality
apply(train[, factor_cols_train], 2, function(x) length(unique(x)))

#none of the factors have high cardinality except ps_car_11_cat which has 104
#this should only be a problem for randomForest

write_csv(train, 'train_clean.csv')
write_csv(test, 'test_clean.csv')


#----------------------------------------------build the linear model--------------------------------------


#set seed for reproducibiliy 
set.seed(10)

#Subset to validate prediction models later
train_indices <- sample(1:nrow(train),size=floor(nrow(train)/2)) 
validate <- train[-train_indices,]  #Set aside subset for validation

#build initial logisitc regression using all of the variables (except id) 
glm.train<-glm(target~.-id,data=train[train_indices,],family =binomial)

#investigate model to check which variables are statistically significant 
summary(glm.train)


#retrain model using only varibles that were significant at the *.05 leve. 
glm.train.2<-glm(target~ps_ind_01+ps_ind_02_cat+ps_ind_03+ps_ind_04_cat+ps_ind_05_cat
                 +ps_ind_07_bin+ps_ind_08_bin+ps_ind_15+ps_ind_16_bin+ps_ind_17_bin
                 +ps_reg_01+ps_reg_02+ps_car_01_cat+ps_car_04_cat+ps_car_06_cat+ps_car_07_cat
                 +ps_car_09_cat+ps_car_11_cat+ps_car_12+ps_car_13+ps_car_14 +ps_car_15 
                 +ps_calc_01
                 ,data=train[train_indices,],family =binomial)

#investigate model again to see is all of the variables used in the logistic regression are stil significant 
summary(glm.train.2)

#all variables are significant at the .05 level


#with a p value of .05533  ps_ind_18_bin was close to being significant. Lets build another model
#with it incluced and use cross validation and test performance to determine which is better 

glm.train.3<-glm(target~ps_ind_01+ps_ind_02_cat+ps_ind_03+ps_ind_04_cat+ps_ind_05_cat
                 +ps_ind_07_bin+ps_ind_08_bin+ps_ind_15+ps_ind_16_bin+ps_ind_17_bin
                 +ps_reg_01+ps_reg_02+ps_car_01_cat+ps_car_04_cat+ps_car_06_cat+ps_car_07_cat
                 +ps_car_09_cat+ps_car_11_cat+ps_car_12+ps_car_13+ps_car_14 +ps_car_15 
                 +ps_calc_01+ps_ind_18_bin 
                 ,data=train[train_indices,],family =binomial)

#investigate model again to see is all of the variables used in the logistic regression are stil significant 
summary(glm.train.3)


#all remaining variables are significant at the .05 level with the exception of ps_ind_18_bin at .051920


#compare the performance of glm.train.2 and glm.train.3 using k-fold cross-validation with k=10

#cv for glm.train.2
glm.train.2_cv_error=cv.glm(train[train_indices,] ,glm.train.2,K=10)$delta[1]

glm.train.2_cv_error #0.03485311


#cv for glm.train.3
glm.train.3_cv_error=cv.glm(train[train_indices,] ,glm.train.3,K=10)$delta[1]

glm.train.3_cv_error#0.03484904

#cv error is very close between the 2 models, but glm.train.3 is slightly better 

#Generally we would evaluate the performance on the held out validation set, but since the classes are so imblanced 
# there is not a natural threshold like .5 to convert probabilities into prediction so that the test error classification rate can  be compared 
#
#In general creating a confusion matrix for the logisic regression would look like:
#intialize vector of 0's to serve as the classification prediction
# glm.pred.2=rep(0,nrow(validate))
# glm.pred.2[glm.train.2.probs>.5]=1
# 
# table(glm.pred.2,validate$target)

#retain glm.train.3 on the entire traing data set. 

glm.train.3.entire<-glm(target~ps_ind_01+ps_ind_02_cat+ps_ind_03+ps_ind_04_cat+ps_ind_05_cat
                 +ps_ind_07_bin+ps_ind_08_bin+ps_ind_15+ps_ind_16_bin+ps_ind_17_bin
                 +ps_reg_01+ps_reg_02+ps_car_01_cat+ps_car_04_cat+ps_car_06_cat+ps_car_07_cat
                 +ps_car_09_cat+ps_car_11_cat+ps_car_12+ps_car_13+ps_car_14 +ps_car_15 
                 +ps_calc_01+ps_ind_18_bin 
                 ,data=train,family =binomial)



#make predictions on the test set 
test.probs=predict(glm.train.3.entire,newdata =test ,type ="response")



#prevent r from using scientific notation since the submission id's must be integers
options(scipen = 999)

#create a data fram with the ids and the predicted probabilities
predictions<-as.data.frame(cbind(id=test$id,target=test.probs))

#write the data frame with the test predictions to a csv  
write.csv(predictions, file = "linear_safe_drive_4.csv",row.names=FALSE)
