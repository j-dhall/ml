---
layout: post
title: "Iris Flower Species Classification using Gaussian Naive Bayes"
date: 2021-02-17 21:55:59 +0530
categories:
---


## <font color="orange">Gaussian Naive Bayes for Iris Flower Species Classification</font>

### Imports


```python
import numpy as np
import pandas as pd #input
from numpy.random import rand
from numpy import mean, std #mean and standard deviation for gaussian probabilities
from scipy.stats import norm #gaussian probabilities
from math import log # to calculate posterior probability
```

### Constants


```python
class_colname = 'class'
train_ds_percent = 0.8
```

### Environment

#### Iris Flower Species


```python
f_data = '../input/iris-species/Iris.csv'
f_cols = ['SepalLengthCm',  'SepalWidthCm',  'PetalLengthCm',  'PetalWidthCm', 'Species']
```

#### Machine Learning Mastery

<img src="{{site.baseurl}}/assets/images/MLMastery-GaussianNB.png">

f_data = '../input/ml_mastery/MLMastery-GaussianNB.csv'
f_cols = ['X1', 'X2', 'Y']

### Data

#### read the csv file


```python
#read the csv file
df = pd.read_csv(f_data)
```

#### drop unwanted columns


```python
#drop unwanted columns
drop_cols = list(set(df.columns) - set(f_cols))
df = df.drop(drop_cols, axis = 1)
```

#### rename last column that supposedly has a class/label


```python
#rename the last column to 'class'
cols = df.columns.to_list()
cols[len(cols)-1] = class_colname
df.columns = cols
```

#### Sanity check for data getting loaded


```python
print(df.head(2))
```

       SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm        class
    0            5.1           3.5            1.4           0.2  Iris-setosa
    1            4.9           3.0            1.4           0.2  Iris-setosa
    

### Model

#### Training Algorithm


```python
def train_gaussian_nb(df, class_colname='class'):
    #number of classes
    classes = df[class_colname].unique()
    num_classes = len(df[class_colname].unique())
    #number of features
    features = df.columns[:-1]
    num_features = len(features)
    #number of data points
    N = len(df)
    
    #data structures for priors and
    # (mean, standard deviation) pairs for each feature and class
    # to later calculate likelihood (conditional probability of feature given class)
    prior = np.zeros(num_classes)
    mean_std = np.zeros((num_classes, num_features, 2), dtype=float)
    
    #for each class...
    for cls in range(num_classes):
        #calculate prior probability of data point belonging to class cls
        prior[cls] = len(df[df[class_colname]==classes[cls]]) / N

        #to later calculate likelihood: conditional probability for all features, given class cls,
        #we store the mean and standard deviation of all features, given class cls
        for i_feature in range(num_features):
            #store mean for i_feature, given cls
            mean_std[cls][i_feature][0] = mean(df[df[class_colname]==classes[cls]].iloc[:, i_feature])
            #store standard deviation for i_feature, given cls
            mean_std[cls][i_feature][1] = std(df[df[class_colname]==classes[cls]].iloc[:, i_feature])
            
    return prior, mean_std, classes, features
```

#### Prediction Algorithm


```python
def apply_gaussian_naive_bayes(num_classes, num_features, prior, mean_std, x):
    score = np.zeros((num_classes), dtype=float)
    
    #for each class...
    for cls in range(num_classes):
        #print('class:', cls)
        
        #for this class, add the log-prior probability to the score
        score[cls] += log(prior[cls], 10) #log to the base 10
        
        #for each feature, add the log-likelihood to the score
        for i_feature in range(num_features):
            #print('feature', i_feature)
            #calculate likelihood from the trained mean and standard deviation
            mu = mean_std[cls][i_feature][0]
            sigma = mean_std[cls][i_feature][1]
            likelihood = norm(mu, sigma).pdf(x[i_feature])
            #add the log-likelihood to the score
            score[cls] += log(likelihood, 10) #log to the base 10
    
    #return the index of class with the maximum-a-posterior probability
    return score.argmax()
```

#### Training

##### split dataset into training and testing


```python
mask = rand(len(df))<train_ds_percent
df_train = df[mask]
df_test = df[~mask]
```

##### train


```python
prior, mean_std, classes, features = train_gaussian_nb(df_train, class_colname)
```

#### Prediction


```python
count_correct, count_incorrect = 0, 0
for index, row in df_test.iterrows():
    actual_cls = row[class_colname]
    pred_cls = apply_gaussian_naive_bayes(len(classes), len(features), prior, mean_std, row[:-1].to_list())
    if classes[pred_cls] == actual_cls:
        count_correct += 1
    else:
        count_incorrect += 1
    #print('(predicted, actual):', classes[pred_cls], row[class_colname])
print('Correct: ', count_correct, 'Incorrect: ', count_incorrect)
print('Percentage of correct predictions: ', (count_correct * 100)/(count_correct + count_incorrect))
```

    Correct:  62 Incorrect:  4
    Percentage of correct predictions:  93.93939393939394
    
