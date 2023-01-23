Machine Learning Project
================

## Data Cleaning

``` r
library(rpart)
library(caret)
library(pROC)
library(tidyverse)
library(randomForest)
training <- read.csv("pml-training.csv",na.strings=c("NA","#DIV/0!", ""))
testing <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!", ""))
training <- training[,colSums(is.na(training)) == 0]
testing <- testing[,colSums(is.na(testing)) == 0]
training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]
```

## Train-Test Split

``` r
set.seed(1)
idx <- createDataPartition(training$classe, p=0.7, list=FALSE)
train <- training[idx,]
test <- training[-idx,]
```

## Decision Tree

``` r
mod1 <- rpart(classe ~ ., data=train, method="class")
result1 <- predict(mod1, test,type = 'class')
confusionMatrix(result1, as.factor(test$classe))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1554  230   16   73   36
    ##          B   30  564   47   14   60
    ##          C   53  150  844  109  109
    ##          D   20   81   81  669   92
    ##          E   17  114   38   99  785
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7504          
    ##                  95% CI : (0.7391, 0.7614)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6831          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9283  0.49517   0.8226   0.6940   0.7255
    ## Specificity            0.9157  0.96818   0.9134   0.9443   0.9442
    ## Pos Pred Value         0.8140  0.78881   0.6672   0.7094   0.7455
    ## Neg Pred Value         0.9698  0.88878   0.9606   0.9403   0.9385
    ## Prevalence             0.2845  0.19354   0.1743   0.1638   0.1839
    ## Detection Rate         0.2641  0.09584   0.1434   0.1137   0.1334
    ## Detection Prevalence   0.3244  0.12150   0.2150   0.1602   0.1789
    ## Balanced Accuracy      0.9220  0.73168   0.8680   0.8192   0.8349

We can see from the result that the accuracy for decision tree algorithm
is 0.7504 with 95% CI of (0.7391, 0.7614).

## Random Forest

``` r
mod2 <- randomForest(as.factor(classe) ~. , data=train, method="class")
result2 <- predict(mod2, test, type = "class")
confusionMatrix(result2, as.factor(test$classe))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1672    1    0    0    0
    ##          B    2 1138   12    0    0
    ##          C    0    0 1009   15    0
    ##          D    0    0    5  948    2
    ##          E    0    0    0    1 1080
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9935          
    ##                  95% CI : (0.9911, 0.9954)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9918          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9988   0.9991   0.9834   0.9834   0.9982
    ## Specificity            0.9998   0.9971   0.9969   0.9986   0.9998
    ## Pos Pred Value         0.9994   0.9878   0.9854   0.9927   0.9991
    ## Neg Pred Value         0.9995   0.9998   0.9965   0.9968   0.9996
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2841   0.1934   0.1715   0.1611   0.1835
    ## Detection Prevalence   0.2843   0.1958   0.1740   0.1623   0.1837
    ## Balanced Accuracy      0.9993   0.9981   0.9902   0.9910   0.9990

We can see from the result that the accuracy for random forest algorithm
is 0.9941 with 95% CI of (0.9917, 0.9959).

## Final Prediction

We will choose the random forest algorithm since it has a better
prediction result than the decision tree algorithm. And here is the
final prediction for the 20 test cases.

``` r
final <- predict(mod2, testing, type="class")
final
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
