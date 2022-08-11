

## Added Required Libraries

library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)

## Download the Data
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

## Read the Data
train_raw_data <- read.csv("./coursera_practical_Learning_project/pml-training.csv")
test_raw_data <- read.csv("./coursera_practical_Learning_project/pml-testing.csv")
dim(train_raw_data)
dim(test_raw_data)


# Clean the data
sum(complete.cases(train_raw_data))

## First, we remove columns that contain NA missing values.
train_raw_data <- train_raw_data[, colSums(is.na(train_raw_data)) == 0]
test_raw_data <- test_raw_data[, colSums(is.na(test_raw_data)) == 0]
summary(train_raw_data)
summary(test_raw_data)

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
# Train & test split the data

set.seed(22519) # For reproducibile purpose
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
trainDataset <- trainCleaned[inTrain, ]
testDataset <- trainCleaned[-inTrain, ]
dim(trainDataset)
dim(testDataset)

# Data Modeling

controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainDataset, method="rf", trControl=controlRf, ntree=250)
modelRf
predictRf <- predict(modelRf, testDataset)
confusionMatrix(testDataset$classe, predictRf)

# Find out the Accuracy

accuracy <- postResample(predictRf, testDataset$classe)
accuracy
oose <- 1 - as.numeric(confusionMatrix(testDataset$classe, predictRf)$overall[1])
oose

# Predicting for Test Data Set

result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
result

# Appendix: Figures

## 1.Correlation Matrix Visualization
corrPlot <- cor(trainDataset[, -length(names(trainDataset))])
corrplot(corrPlot, method="color")
## 2.Decision Tree Visualization
treeModel <- rpart(classe ~ ., data=trainDataset, method="class")
prp(treeModel) # Creating fast plot


