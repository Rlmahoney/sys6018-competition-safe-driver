# SYS6018
# Kaggle Competition 4
# Group 4-3
# Yi Hao , Robert Mahoney , Duncan Rule
# yh8a,



install.packages("tidyverse")
library(readr)  
library(dplyr)  
library(ggplot2)
library(tidyverse)


##Read data and data exploration

driver <- read.csv("train_clean.csv")
dim(driver)
     #[1] 595212     57
str(driver)
summary(driver)

sapply(driver, function(x) sum(is.na(x)))
     # no missing value

table(driver$target)
     #   0        1 
     # 573518   21694 

prop.table(table(driver$target))
     #          0              1 
     # 0.96355248     0.03644752

factorColumns <- c("ps_ind_01","ps_ind_02_cat","ps_ind_03","ps_ind_04_cat","ps_ind_05_cat",
                   
                   "ps_ind_06_bin", "ps_ind_07_bin","ps_ind_08_bin","ps_ind_09_bin","ps_ind_10_bin","ps_ind_11_bin","ps_ind_12_bin",
                   
                   "ps_ind_13_bin","ps_ind_14","ps_ind_15","ps_ind_16_bin","ps_ind_17_bin","ps_ind_18_bin", 
                   
                   "ps_car_01_cat","ps_car_02_cat","ps_car_04_cat","ps_car_06_cat",
                   
                   "ps_car_07_cat","ps_car_08_cat", "ps_car_09_cat", "ps_car_10_cat",
                   
                   "ps_car_11", 
                   
                   "ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04", "ps_calc_05",
                   
                   "ps_calc_06","ps_calc_07","ps_calc_08","ps_calc_09","ps_calc_10",
                   
                   "ps_calc_11","ps_calc_12","ps_calc_13", "ps_calc_14",
                   
                   "ps_calc_15_bin","ps_calc_16_bin","ps_calc_17_bin","ps_calc_18_bin","ps_calc_19_bin", "ps_calc_20_bin")


numericColumns = c("ps_reg_03","ps_car_11_cat", "ps_car_12", "ps_car_13", "ps_car_14")


# "ps_car_11_cat" was removed from this factor list, because of too many levels.

driver[factorColumns] <- lapply(driver[factorColumns], as.factor) 

driver[numericColumns] <- lapply(driver[numericColumns], as.numeric)

str(driver)

#na_count <-sapply(driver, function(y) sum(length(which(is.na(y))))); na_count
#apply(df, 2, function(x) any(is.na(x)))
#sapply(df, function(x) sum(is.na(x)))
#apply(driver, 2, function(x) any(is.na(x)))


##Data Partition
set.seed(777)
s<- sample(1:2, nrow(driver), replace= TRUE, prob = c(0.7,0.3))
train <- driver[s==1,]
test <- driver[s==2,]

table(train$target)
        #      0        1 
        # 400991    15368 

prop.table(table(train$target))
        #         0              1 
        #0.96308955     0.03691045 

table(test$target)
        #      0      1 
        # 172527   6326 

prop.table(table(test$target))
        #          0             1 
        # 0.96463017    0.03536983 

train$target = as.factor(train$target)
test$target = as.factor(test$target)



#########################################
#                                       #
#     Subsetting of imbalanced data     #
#                                       #
#########################################

set.seed(777)

train1 = subset(train, target==1)
train0 = subset(train, target==0)

nrow(train1)    #[1] 15368
nrow(train0)    #[1] 400991

#Keep the train1, and subset the train0 into 6 data set of size of 4 * size(train1). 4*15368= 61472
#Subset 
sub = sample(1:nrow(train0), size = 61472)  
train01 = train0[sub, ]
train01r = train0[-sub, ]

sub = sample(1:nrow(train01r), size = 61472)
train02 = train01r[sub, ]
train02r = train01r[-sub, ]

sub = sample(1:nrow(train02r), size = 61472)
train03 = train02r[sub, ]
train03r = train02r[-sub, ]

sub = sample(1:nrow(train03r), size = 61472)
train04 = train03r[sub, ]
train04r = train03r[-sub, ]

sub = sample(1:nrow(train04r), size = 61472)
train05 = train04r[sub, ]
train05r = train04r[-sub, ]

sub = sample(1:nrow(train05r), size = 61472)
train06 = train05r[sub, ]
train06r = train05r[-sub, ]


#Merge the train01-06 sets with train1

train1f = rbind(train01, train1)
train2f = rbind(train02, train1)
train3f = rbind(train03, train1)
train4f = rbind(train04, train1)
train5f = rbind(train05, train1)
train6f = rbind(train06, train1)


### Fit Random Forest Models Using train1f data

install.packages("MASS")
install.packages("randomForest")
library(MASS)
library(randomForest)

set.seed(777)
train01$target <- as.factor(train01$target)
test$target <- as.factor(test$target)


#Fit random Forest Model:

# train1f training set
dim(train1f)

Sys.time() 
rf1.mod <-randomForest(target~.-id, data= train1f, ntree= 200)
Sys.time() 
print(rf1.mod)
plot(rf1.mod)



## ----- Try different Sampling methods ------- ##

##install packages
install.packages("ROSE")
install.packages("rpart")
library(ROSE)
library(rpart)

#train1f training set
dim(train1f)

#Over sampling
train1f_over <- ovun.sample(target ~.-id, data = train1f, method = "over", N = 61472*2)$data
table(train1f_over$target)

#Under sampling
train1f_under <- ovun.sample(target ~.-id, data = train1f, method = "under", N = 15368*2)$data
table(train1f_under$target)

#Both sampling
train1f_both <- ovun.sample(target ~.-id, data = train1f, method = "both", N=76840, seed=1)$data
table(train1f_both$target)

#ROSE to generate data synthetically
train1f.rose <- ROSE(target~.-id, data = train1f, seed = 1)$data
table(train1f.rose$target)



##Fit Random Forest Models

rf1.mod <-randomForest(target~.-id, data= train1f, ntree= 200)
rf1.mod1 <-randomForest(target~.-id, data= train1f_over, ntree= 200)
rf1.mod2 <-randomForest(target~.-id, data= train1f_under, ntree= 200)
rf1.mod3 <-randomForest(target~.-id, data= train1f_both, ntree= 200)
rf1.mod4 <-randomForest(target~.-id, data= train1f.rose, ntree= 200)

print(rf1.mod)
print(rf1.mod1)
print(rf1.mod2)
print(rf1.mod3)
print(rf1.mod4)

plot(rf1.mod)
plot(rf1.mod1)
plot(rf1.mod2)
plot(rf1.mod3)
plot(rf1.mod4)


#Predict and Confusion Matrix ---- Train
library(caret)
p <- predict(rf1.mod, data= train1f)
p1 <- predict(rf1.mod1, data= train1f)
p2 <- predict(rf1.mod2, data= train1f)
p3 <- predict(rf1.mod3, data= train1f)
p4 <- predict(rf1.mod4, data= train1f)

#Confusion Matrix
confusionMatrix(p, train1f$target)
confusionMatrix(p1, train1f_over$target)
confusionMatrix(p2, train1f_under$target)
confusionMatrix(p3, train1f_both$target)
confusionMatrix(p4, train1f.rose$target)


##Predict and Confusion Matrix ---- Test
set.seed(777)
p1t <- predict(rf1.mod, newdata= test)
p1t1 <- predict(rf1.mod1, newdata= test)
p1t2 <- predict(rf1.mod2, newdata= test)
p1t3 <- predict(rf1.mod3, newdata= test)
p1t4 <- predict(rf1.mod4, newdata= test)


confusionMatrix(p1t, test$target)
confusionMatrix(p1t1, test$target)
confusionMatrix(p1t2, test$target)
confusionMatrix(p1t3, test$target)
confusionMatrix(p1t4, test$target)


#Tune random forest model:

#turn mtry

t <- tuneRF(train1f [, -c(1,2)], train1f[, 2], stepFactor = 0.5, plot = T, ntreeTry = 50, trace = T, improve = 0.05)
t1 <- tuneRF(train1f_over [, -c(1,2)], train1f_over[, 2], stepFactor = 0.5, plot = T, ntreeTry = 50, trace = T, improve = 0.05)
t2 <- tuneRF(train1f_under [, -c(1,2)], train1f_under[, 2], stepFactor = 0.5, plot = T, ntreeTry = 50, trace = T, improve = 0.05)
t3 <- tuneRF(train1f_both [, -c(1,2)], train1f_both[, 2], stepFactor = 0.5, plot = T, ntreeTry = 50, trace = T, improve = 0.05)
t4 <- tuneRF(train1f.rose [, -c(1,2)], train1f.rose[, 2], stepFactor = 0.5, plot = T, ntreeTry = 50, trace = T, improve = 0.05)


##Re-Fit Random Forest Models 
Sys.time()
rf1t.mod <-randomForest(target~.-id, data= train1f, ntree= 200, mtry=3)
Sys.time()
rf1t.mod1 <-randomForest(target~.-id, data= train1f_over, ntree= 200, mtry = 14)
rf1t.mod2 <-randomForest(target~.-id, data= train1f_under, ntree= 200, mtry =14)
rf1t.mod3 <-randomForest(target~.-id, data= train1f_both, ntree= 200, mtry=14)
rf1t.mod4 <-randomForest(target~.-id, data= train1f.rose, ntree= 200, mtry=14)

print(rf1t.mod)
print(rf1t.mod1)
print(rf1t.mod2)
print(rf1t.mod3)
print(rf1t.mod4)

plot(rf1t.mod)
plot(rf1t.mod1)
plot(rf1t.mod2)
plot(rf1t.mod3)
plot(rf1t.mod4)


#Predict and Confusion Matrix ---- Train
library(caret)
pt <- predict(rf1t.mod, data= train1f)
p1t <- predict(rf1t.mod1, data= train1f)
p2t <- predict(rf1t.mod2, data= train1f)
p3t <- predict(rf1t.mod3, data= train1f)
p4t <- predict(rf1t.mod4, data= train1f)

#Confusion Matrix
cm = confusionMatrix(pt, train1f$target)
confusionMatrix(p1t, train1f_over$target)
confusionMatrix(p2t, train1f_under$target)
confusionMatrix(p3t, train1f_both$target)
confusionMatrix(p4t, train1f.rose$target)



##Predict and Confusion Matrix ---- Test
set.seed(777)
p1tt <- predict(rf1t.mod, newdata= test)
p1t1t <- predict(rf1t.mod1, newdata= test)
p1t2t <- predict(rf1t.mod2, newdata= test)
p1t3t <- predict(rf1t.mod3, newdata= test)
p1t4t <- predict(rf1t.mod4, newdata= test)


confusionMatrix(p1tt, test$target)
confusionMatrix(p1t1t, test$target)
confusionMatrix(p1t2t, test$target)
confusionMatrix(p1t3t, test$target)
confusionMatrix(p1t4t, test$target)

## Models are selected by Out Of Bag (OOB) errors.
## Sensitivity, specificity and misclassification are also considered.
## Pick model rf1t.mod1 and rf1t.mod3 

##Check the cutoff point:
install.packages("ROCR")
library(ROCR)

pred1 <- predict(rf1t.mod1, test, type = "prob")
pred3 <- predict(rf1t.mod3, test, type = "prob")
hist(pred1)
hist(pred3)
pred1 <- prediction(pred1[ ,2], test$target)
pred3 <- prediction(pred3[ ,2], test$target)
eval1 <- performance(pred1, 'acc')
eval3 <- performance(pred3, 'acc')
plot(eval1)
plot(eval3)

#Identify the Best Cutoff and Accuracy
max1 <- which.max(slot(eval1, 'y.values')[[1]])
max3 <- which.max(slot(eval3, 'y.values')[[1]])

acc1 <- slot(eval1, 'y.values')[[1]][max1]
acc3 <- slot(eval3, 'y.values')[[1]][max3]

cut1 <- slot(eval1, 'x.values')[[1]][max1]
cut3 <- slot(eval3, 'x.values')[[1]][max3]

print(c(Accuracy = acc1, Cutoff = cut1))
print(c(Accuracy = acc3, Cutoff = cut3))


##Refit model rf1.mod1 using new cutoff
#rf1tc.mod1 <-randomForest(target~.-id, data= train1f_over, ntree= 200, mtry = 14, cutoff = c(0.7, 0.3))
##Predict and Confusion Matrix ---- Test
#set.seed(777)
#p1t1tc <- predict(rf1tc.mod1, newdata= test)
#confusionMatrix(p1t1tc, test$target)


##### Fit the final model using whole driver set

#Problem was that oversampling of whole train data generated too large data and the computer failed to run. 
#The used both-sampling method, but it lost some factor levels in target= 0 population.
#So here used the original whole train set.

#driver_over <- ovun.sample(target ~.-id, data = driver, method = "both", N = 595212/2)$data
#table(driver_over$target)
#      0      1 
# 297462 297750 

dim(driver)
table(driver$target)
#   0        1 
# 573518   21694 

#Fitting the final model using the orignial whole train set --driver:

rf1w.mod <-randomForest(target~.-id, data= driver, ntree= 200, mtry = 14)

print(rf1w.mod)

plot(rf1w.mod)

## Prediction of safe driver

ptest <- read.csv("test_clean.csv")

dim(ptest)
          #892816     56
str(ptest)

sapply(ptest, function(x) sum(is.na(x)))
# no missing value

ptest[factorColumns] <- lapply(ptest[factorColumns], as.factor) 

ptest[numericColumns] <- lapply(ptest[numericColumns], as.numeric)

str(ptest)

#prediction of target

pred1w <- predict(rf1w.mod1, newdata= ptest, type = "prob")


#submission files
id<- ptest$id
target<- pred1w
submission <- cbind(id, target)
submission<-as.data.frame(submission)


# Export to csv
write.csv(submission1, file = "Competition_4-3_Safe_Driver1.csv", row.names = F)


