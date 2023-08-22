---
title: "Prostate Cancer Detection: Applying kNN Algorithm for Accurate Diagnosis"
author: "Sheetal Nighut"
date: "10/4/2023"
output: html_document
---

Project - Improving Prostate Cancer Diagnosis Accuracy: Implementing k-Nearest Neighbors Algorithm for Precise Detection

Download the data set: Prostate_Cancer.csv data set, which contains information about patients with prostate cancer - 100 rows and 10 column.

```{r}
prc <- read.csv("Prostate_Cancer.csv",stringsAsFactors = FALSE)

# check the string
str(prc)
dim(prc)

```
The variables include 'id', 'diagnosis_result', 'radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'symmetry', and 'fractal_dimension

Data preprocessing and exploration:

```{r}
#removes variable(id)
prc <- prc[-1] 

# get count the numbers of patients
table(prc$diagnosis_result)
```
Diagnosis Result Proportions:

```{r}
# diagnosis_result variable as factor with labels
prc$diagnosis_result <- factor(prc$diagnosis_result, levels = c("B", "M"), labels = c("Benign", "Malignant"))

# the percentage form rounded of to 1 decimal place
round(prop.table(table(prc$diagnosis)) * 100, digits = 1) 
```


```{r}
# normalize function
normalize <- function(x) { return ((x - min(x)) / (max(x) - min(x))) }

# save as data frame
prc_n <- as.data.frame(lapply(prc[2:9], normalize))

#check summary
summary(prc_n$radius)
```
Splitting the data into training and testing sets

```{r}

# Splitting the normalized data

#divide the prc_n data frame into prc_train and prc_test data frames
prc_train <- prc_n[1:65,] 
prc_test <- prc_n[66:100,]

# Creating labels for the training and testing sets:

#This code takes the diagnosis factor in column 1 of the prc data frame and on turn creates prc_train_labels and prc_test_labels data frame.
prc_train_labels <- prc[1:65, 1]
prc_test_labels <- prc[66:100, 1]
```

Training and prediction using kNN:
kNN algorithm is used to predict the labels for the test dataset using the trained model and the specified number of nearest neighbors (k).
```{r}
# Training a model on data

library(class)

# perform the kNN algorithm
prc_test_pred <- knn(train = prc_train, test = prc_test, cl = prc_train_labels, k=10)
```

Generate a contingency table and perform cross-tabulation between the predicted labels (prc_test_pred) and the actual labels (prc_test_labels)

```{r}

#install.packages("gmodels")
library(gmodels)

# call the CrossTable function
crstbl <- CrossTable(x = prc_test_labels, y = prc_test_pred, prop.chisq = FALSE)

# view results in contingency table
crstbl$t
```

Results highlights that the kNN algorithm achieved higher accuracy in predicting the 'Malignant' label compared to the 'Benign' label. It correctly identified all instances of 'Malignant' but had some misclassifications in predicting 'Benign' cases. Further analysis and refinement of the model may be required to improve its performance on the 'Benign' label predictions.


```{r}

# Determine TN (true negatives), which corresponds to the number of observations that are correctly predicted as the negative class i. e Benign
TN <- crstbl$t[1,1]

# Determine TP (true positives), which corresponds to the number of observations that are correctly predicted as the positive class i.e Malignant
TP <- crstbl$t[2,2]

# calculate the accuracy of the model
((TN+TP)/35) *100 
```

Trying another kNN implementation from another package, such as the caret package. Compare the accuracy of the two implementations.

```{r}

library(ggplot2)
library(caret)

#set seed value for reproducibility
set.seed(1)

# creating data partition for training and testing purposes with a 75% proportion for the training set and the remaining 25% for the testing set.
inTrainingSet <- createDataPartition(y = prc$diagnosis_result, p=0.75, list = FALSE)

# random split of dataset by subsetting the prc dataset 
diag_train <- prc[inTrainingSet,]
diag_test <- prc[-inTrainingSet,]

# generates a table of the proportions of each category in the diagnosis_result variable in the prc dataset
round(prop.table(table(prc$diagnosis_result))*100, 1)
```

```{r}
# calculate the percentage distribution of the categories 
round(prop.table(table(diag_train$diagnosis_result))*100, 1)
```


```{r}
# calculate the percentage distribution of the categories 
round(prop.table(table(diag_test$diagnosis_result))*100, 1)
```

```{r}
# use trainControl for resampling method and number of repetitions.
ctrl <- trainControl(method="repeatedcv",repeats = 3) 

#knFit using train function for model training process
knnFit <- train(diagnosis_result ~., data = diag_train, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)

# view results 
knnFit
```

The trained k-Nearest Neighbors model achieved an accuracy of approximately 84.5% on the training data
```{r}
#Plotting yields Number of Neighbours Vs accuracy (based on repeated cross validation)
plot(knnFit)
```

