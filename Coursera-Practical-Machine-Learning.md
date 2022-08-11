---
title: "practical-Machine-Learning"
author: "Kaushik Prasad Dey"
date: "8/11/2022"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.

## Added Required Libraries

```{r library loaded}
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
```

## Download the Data

```{r download the data}
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- "./coursera_practical_Learning_project/pml-training.csv"
testFile  <- "./coursera_practical_Learning_project/pml-testing.csv"
if (!file.exists("./coursera_practical_Learning_project")) {
  dir.create("./coursera_practical_Learning_project")
}
if (!file.exists(trainFile)) {
  download.file(trainUrl, destfile=trainFile, method="curl")
}
if (!file.exists(testFile)) {
  download.file(testUrl, destfile=testFile, method="curl")
}
```

## Read the Data
```{r read the data}
train_raw_data <- read.csv("./coursera_practical_Learning_project/pml-training.csv")
test_raw_data <- read.csv("./coursera_practical_Learning_project/pml-testing.csv")
dim(train_raw_data)
dim(test_raw_data)
```
The training data set contains 19622 observations and 160 variables, while the testing data set contains 20 observations and 160 variables. The "classe" variable in the training set is the outcome to predict.

# Clean the data

In this step, we will clean the data and get rid of observations with missing values as well as some meaningless variables.

```{r clean the data}
sum(complete.cases(train_raw_data))
```
## First, we remove columns that contain NA missing values.

```{r Treatment for NA missing Values}
train_raw_data <- train_raw_data[, colSums(is.na(train_raw_data)) == 0]
test_raw_data <- test_raw_data[, colSums(is.na(test_raw_data)) == 0]
summary(train_raw_data)
summary(test_raw_data)
```

Next, we get rid of some columns that do not contribute much to the accelerometer measurements.

```{r Get Rid of some columns}
classe <- train_raw_data$classe
trainRemove <- grepl("^X|timestamp|window", names(train_raw_data))
train_raw_data <- train_raw_data[, !trainRemove]
trainCleaned <- train_raw_data[, sapply(train_raw_data, is.numeric)]
trainCleaned$classe <- classe

testRemove <- grepl("^X|timestamp|window", names(test_raw_data))
test_raw_data <- test_raw_data[, !testRemove]
testCleaned <- test_raw_data[, sapply(test_raw_data, is.numeric)]
dim(trainCleaned)
dim(testCleaned)
```
Now, the cleaned training data set contains **(19622)** observations and **(53)** variables, while the testing data set contains **(20)** observations and **(53)** variables. The "classe" variable is still in the cleaned training set.

# Train & test split the data

Then, we can split the cleaned training set into a pure training data set **(70%)** and a validation data set **(30%)**. We will use the validation data set to conduct cross validation in future steps.

```{r Slicing the Data}
set.seed(22519) # For reproducibile purpose
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
trainDataset <- trainCleaned[inTrain, ]
testDataset <- trainCleaned[-inTrain, ]
dim(trainDataset)
dim(testDataset)
```

# Data Modeling

We fit a predictive model for activity recognition using **Random Forest** algorithm because it automatically selects important variables and is robust to correlated covariates & outliers in general. We will use **5-fold cross validation** when applying the algorithm.

```{r training and testing Model Creation}
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainDataset, method="rf", trControl=controlRf, ntree=250)
modelRf
```

Then, we estimate the performance of the model on the validation data set.

```{r Confusion Matrix Creation}
predictRf <- predict(modelRf, testDataset)
confusionMatrix(testDataset$classe, predictRf)
```

# Find out the Accuracy

```{r Find out the accuracy}
accuracy <- postResample(predictRf, testDataset$classe)
accuracy
```

```{r Find out the numeric Confusion Matrix}
oose <- 1 - as.numeric(confusionMatrix(testDataset$classe, predictRf)$overall[1])
oose
```

So, the estimated accuracy of the model is **99.30%** and the estimated out-of-sample error is **0.70%**.

# Predicting for Test Data Set

Now, we apply the model to the **original testing data** set downloaded from the data source. 
**We remove the problem_id column first** .

```{r Predicting for Test Data Set}
result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
result
```

# Appendix: Figures

   ## 1.Correlation Matrix Visualization
  
```{r Maing Co-relation plot creation}
corrPlot <- cor(trainDataset[, -length(names(trainDataset))])
corrplot(corrPlot, method="color")
```
   ## 2.Decision Tree Visualization
   
```{r Maing Decesion Tree Visualization}
treeModel <- rpart(classe ~ ., data=trainDataset, method="class")
prp(treeModel) # Creating fast plot
```


