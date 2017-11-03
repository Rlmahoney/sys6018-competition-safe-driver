#Sys 6018
#wdr5ft
#safe driver cleaning

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

#Subset to validate prediction models later
train_indices <- sample(1:nrow(train),size=floor(nrow(train)/2)) 
#train.2 <- train[train_indices,]     #Select subset for training
validate <- train[-train_indices,]  #Set aside subset for validation

#build initial logisitc regression using all of the variables (except id) 
glm.train<-glm(target~.-id,data=train[train_indices,],family =binomial)

#investigate model to check which variables are statistically significant 
summary(glm.train)


#retrain model using only varibles that were significant at the *.05 level 
glm.train.2<-glm(target~ps_ind_01+ps_ind_02_cat+ps_ind_03+ps_ind_05_cat+ps_ind_07_bin+ps_ind_08_bin
                 +ps_ind_15+ps_ind_16_bin+ps_ind_17_bin+ps_reg_01+ps_reg_02+ps_reg_03+ps_car_01_cat+ps_car_04_cat
                 +ps_car_07_cat+ps_car_09_cat+ps_car_11_cat+ps_car_12+ps_car_13+ps_car_14+ps_car_15 
                 ,data=train[train_indices,],family =binomial)

#investigate model again to see is all of the variables used in the logistic regression are stil significant 
summary(glm.train.2)


# ps_ind_16_bin not significant at the .05 level. Rerun the regression without it. 

glm.train.3<-glm(target~ps_ind_01+ps_ind_02_cat+ps_ind_03+ps_ind_05_cat+ps_ind_07_bin+ps_ind_08_bin
                 +ps_ind_15+ps_ind_17_bin+ps_reg_01+ps_reg_02+ps_reg_03+ps_car_01_cat+ps_car_04_cat
                 +ps_car_07_cat+ps_car_09_cat+ps_car_11_cat+ps_car_12+ps_car_13+ps_car_14+ps_car_15 
                 ,data=train[train_indices,],family =binomial)

#investigate model again to see is all of the variables used in the logistic regression are stil significant 
summary(glm.train.3)


#all remaining variables are significant at the .05 level 

#use the logistic regression model to make predictions on the portion of the held of training data

glm.probs =predict (glm.train.3,newdata =validate ,type ="response")


#retain model on the entire set of training data
glm.train.5<-glm(target~ps_ind_01+ps_ind_02_cat+ps_ind_03+ps_ind_05_cat+ps_ind_07_bin+ps_ind_08_bin
                 +ps_ind_15+ps_ind_17_bin+ps_reg_01+ps_reg_02+ps_reg_03+ps_car_01_cat+ps_car_04_cat
                 +ps_car_07_cat+ps_car_09_cat+ps_car_11_cat+ps_car_12+ps_car_13+ps_car_14+ps_car_15
                 ,data=train,family =binomial)


#make predictions on the test set 
test.probs=predict(glm.train.5,newdata =test ,type ="response")



#prevent r from using scientific notation since the submission id's must be integers
options(scipen = 999)

#create a data fram with the ids and the predicted probabilities
predictions<-as.data.frame(cbind(id=test$id,target=test.probs))

#write the data frame with the test predictions to a csv  
write.csv(predictions, file = "linear_safe_drive_3.csv",row.names=FALSE)
