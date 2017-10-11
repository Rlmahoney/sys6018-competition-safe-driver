#Install (if necessary) and load the core tidyverse packages: ggplot2, tibble, tidyr, readr, purrr, and dplyr
library(tidyverse) 
library(readr)

#import the data 
train<-read_csv("driver_train.csv")
test<-read_csv("driver_test.csv")

#21694 total claims
sum(train$target)

#portion of claims  0.03644752
claim_portion<-sum(train$target)/nrow(train)



#prevent r from using scientific notation since the submission id's must be integers
options(scipen = 999)

#assign the overall portion of claims as the probability for each id
predictions<-as.data.frame(cbind(id=test$id,target=claim_portion))


write.csv(predictions, file = "rlm4bj_safe_drive.csv",row.names=FALSE)