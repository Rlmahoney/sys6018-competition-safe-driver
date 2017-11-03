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

#joining datasets so factor levels are the same
master = rbind(train[, -2], test)

#convert to factor
master[factor_cols] = lapply(master[factor_cols], factor)

#split datasets
train_2 = master[1:595212, ]
test_2 = master[595213:nrow(master), ]
train_2$target = train$target
train_2$target = as.factor(train_2$target)

#check cardinality
train_cardinality = apply(train_2[, factor_cols], 2, function(x) length(unique(x)))
test_cardinality = apply(test_2[, factor_cols], 2, function(x) length(unique(x)))

extra_factors = train_cardinality - test_cardinality
#no extra factor levels

#none of the factors have high cardinality except ps_car_11_cat which has 104
#this should only be a problem for randomForest

write_csv(train_2, 'train_clean_2.csv')
write_csv(test_2, 'test_clean_2.csv')
