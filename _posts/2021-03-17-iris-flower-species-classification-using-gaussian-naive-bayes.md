---
layout: post
title: "[without library] Iris Flower Species Classification using Gaussian Naive Bayes"
date: 2021-02-15 20:32:59 +0530
categories:
---

# [without library] <font color="orange">Iris Flower Species Classification using Gaussian Naive Bayes</font>

### Based on - [2010] Generative and Discriminative Classifiers : Naive Bayes and Logistic Regression - <font color=magenta>Tom Mitchell</font>

<img src="{{site.baseurl}}/assets/images/iris-gnb/gnb.png">

## Introduction

This notebook implements <font color=blue>Gaussian Naive Bayes</font>. It performs <font color=blue>multi-class classification</font> on Iris Flower Species dataset consisting of four attributes and belonging to three classes. There are 300 data points that we divide into training and test data. The prediction accuracy is <font color=blue>94-100%</font>.

Resources:
- [CMU Qatar Lecture Notes - Naive Bayes - Gianni A. Di Caro](https://web2.qatar.cmu.edu/~gdicaro/10315/lectures/315-F19-6-NaiveBayes.pdf)
- [CMU Machine Learning 10-701 - Pradeep Ravikumar](http://www.cs.cmu.edu/~pradeepr/courses/701/2018-fall/Fall2018Slides/NaiveBayes.pdf)
- [CMU Machine Learning 10-701 - Tom Mitchell](https://www.cs.cmu.edu/~tom/10701_sp11/slides/LR_1-27-2011.pdf)
- [How is Naive Bayes a Linear Classifier? - stats.stackexchange](https://stats.stackexchange.com/questions/142215/how-is-naive-bayes-a-linear-classifier#:~:text=In%20general%20the%20naive%20Bayes,in%20a%20particular%20feature%20space.)
- [Naive Bayes classifier - Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Bernoulli_na%C3%AFve_Bayes)

Towards the end, I have included a <font color=blue>sidebar</font> on the <font color=magenta>comparison of Naive Bayes and Logistic Regression</font>. 

## Taxonomy and Notes

> Probabilistic, Generative

Naive Bayes is a <font color=blue>probabilistic</font> classifier. It learns the underlying <font color=blue>joint distribution P(X, Y)</font> by learning the <font color=blue>likelihood distribution P(X\|Y)</font> (a.k.a __class-conditonal__) and the <font color=blue>prior distribution P(Y)</font> (a.k.a __class-prior__). The distribution learnt can be used to generate data. Hence, it is a <font color=blue>generative</font> classifier. Naive bayes does not model the class decision boundaries, but instead models the distribution of the observed data. In Generative Modeling, we learn a full probabilistic model, the joint distribution P(X, Y). We assume a parametric form of the underlying probability distribution, and estimate those parameters from the observed data.

> Parametric

Naive Bayes is <font color=blue>parametric</font> since the distribution assumed to model the likelihood (say, gaussian, multinomial, or bernoulli) have parameters. Gaussian distribution has parameters for mean and standard deviation.

> Linear

In general, the Naive Bayes classifier is not linear. But, it becomes <font color=blue>linear, if the likelihood is exponential</font>. This is because the __log-likelihood becomes linear in the log-space__. <font color=blue>Gaussian</font>, <font color=blue>Multinomial</font>, or <font color=blue>Bernoulli</font> distributions are all from the exponential family of distributions, so Naive Bayes can be applied to data exhibiting these distributions. Under the assumption of an exponential distribution, Bernoulli Naive Bayes maps to Binomial Logistic Regression, and Multinomial Naive Bayes maps to Multinomial Logistic Regression. 

> Event Model

The assumptions on distributions of features are called the <font color=blue>event model</font> of the Naive Bayes classifier.

> Event Model - Continuous Data (MNIST Digit Recognition, Iris Flower Species Classification)

When dealing with __continuous data__, a typical __assumption__ is that the continuous values associated with each class are distributed according to a <font color=blue>normal (or Gaussian) distribution</font>. The Iris Flower Species dataset has attributes that exhibit gaussian distribution.

When assuming Gaussian class-conditionals, <font color=blue>if all class-conditional gaussians have the same covariance</font>, then the quadratic terms cancel out and we are left with a <font color=blue>linear</font> form. To take an example of MNIST dataset, if we asume that the variance across digits is the same, then we have a linear model.

The __class prior__ distribution has to be a <font color=blue>discrete distribution</font> (say, multinoulli) that distributes the probability of a data point belonging to class among K possible classes.

Note: Sometimes the distribution of class-conditional marginal densities is __far from normal__. In these cases, <font color=blue>kernel density estimation</font> can be used for a more realistic estimate of the marginal densities of each class.

> Event Model - Discrete Data (Text Classification)

For __discrete features__ (document classification), Multinomial and Bernoulli distributions are popular. With a multinomial event model, features represent frequencies of events (say, count of word occurrences). With a bernoulli event model, features represent presence or absence of events (say, presence or absence of words).

> Binary and Multiclass

Naive Bayes can be both <font color=blue>binary and multiclass</font>.

> Summary

In summary, Naive Bayes is:
- Probabilistic
- Generative
- Binary and Multiclass
- Linear
- Parametric

<img src="{{site.baseurl}}/assets/images/iris-gnb/mmap_nb.png">

> Why Naive?

Naive Bayes makes an assumption that the input attributes are independent of each other. This results in a <font color=blue>significant reduction in the number of parameters</font> the model needs to learn. This is because, since each attribute is assumed to be influenced only by the class its data point belongs to, the model only has P(X_i\|Y_k) terms, and no P(X_i\|X_j), P(X_i\|X_j, X_k), etc terms. This is the reason the model is called '<font color=blue>Naive</font>' because it is seldom the case that the input attributes do not influence each other. Still, Naive Bayes has proven to be effective.

Due to the modeling assumptions of attributes being independent, Naive Bayes model introduces <font color=red>a lot more inductive bias</font> as compared to Logistic Regression. This also results in  <font color=green>faster convergence</font> of order O(log N) (N is the number of data points). The fast convergence can perhaps be on  <font color=red>less accurate</font> estimates.

> Algebraically solved Closed-Form

Naive Bayes class-conditional probabilities (<font color=blue>maximum likelihood estimation</font>) can be deduced analytically. Also, the class-priors can be deduced using frequentist methods. So, there is a <font color=blue>closed-form</font> solution to Naive Bayes. And, hence, we <font color=blue>do not need a numerical method like gradient descent</font>.

## Imports


```python
import numpy as np
import pandas as pd #input
from numpy.random import rand
from numpy import mean, std #mean and standard deviation for gaussian probabilities
from scipy.stats import norm #gaussian probabilities
from math import log # to calculate posterior probability
import seaborn as sns #plotting
import matplotlib.pyplot as plt #plotting
%matplotlib inline
```

## Data

### Data Configuration


```python
class_colname = 'Species'
```

#### Iris Flower Species


```python
f_data = '../input/iris-species/Iris.csv'
f_cols = ['SepalLengthCm',  'SepalWidthCm',  'PetalLengthCm',  'PetalWidthCm', 'Species']
```

### Read

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

#### sanity check for data getting loaded


```python
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44</th>
      <td>5.1</td>
      <td>3.8</td>
      <td>1.9</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>87</th>
      <td>6.3</td>
      <td>2.3</td>
      <td>4.4</td>
      <td>1.3</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <th>72</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>4.9</td>
      <td>1.5</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>76</th>
      <td>6.8</td>
      <td>2.8</td>
      <td>4.8</td>
      <td>1.4</td>
      <td>Iris-versicolor</td>
    </tr>
  </tbody>
</table>
</div>



#### visualize

Note that the <font color=magenta>attributes exhibit gaussian distribution</font>.


```python
def plot_features_violin(data):
    plt.figure(figsize=(15,10))
    plt.subplot(2,2,1)
    sns.violinplot(data=data, x='Species',y='PetalLengthCm')
    plt.subplot(2,2,2)
    sns.violinplot(data=data, x='Species',y='PetalWidthCm')
    plt.subplot(2,2,3)
    sns.violinplot(data=data, x='Species',y='SepalLengthCm')
    plt.subplot(2,2,4)
    sns.violinplot(data=data, x='Species',y='SepalWidthCm')
```


```python
plot_features_violin(df)
```


![png]({{site.baseurl}}/assets/images/iris-gnb/output_28_0.png)


## Model

### Model Configuration


```python
train_ds_percent = 0.8
```

### Training Algorithm

<img src="{{site.baseurl}}/assets/images/iris-gnb/gnb_mean_std.png">


```python
'''
    return
            classes: (list) of unique class names in the dataset,
             got from the last column named class_colname.
             
            features: (list) of features (column names) in the dataset.
             this excludes the last column which we expect it to have the class labels.
             
            prior: (1-d array) of dim num_classes
            (prior probability of a set of features belonging to a class)
            
            mean_std: (3-d array) of dim num_classes x num_features x 2 (2: mean and std)
            (mean and standard deviation for all features, given the class)
            
    arguments:
    df: (dataframe) with features and class names (should have a 'class' column in addition to the feature columns).
    class_colname: (string) provide suitable column name otherwise, using the class_colname argument.
'''
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
            
    return classes, features, prior, mean_std
```

### Prediction Algorithm

<a href="https://www.codecogs.com/eqnedit.php?latex=P\left&space;(&space;X_{i}=x|Y=y_{k}&space;\right&space;)=\frac{1}{\sigma&space;_{ik}\sqrt{2\pi&space;}}e^{\frac{-\left&space;(&space;x-\mu&space;_{ik}&space;\right&space;)^{2}}{2\sigma&space;_{ik}^{2}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P\left&space;(&space;X_{i}=x|Y=y_{k}&space;\right&space;)=\frac{1}{\sigma&space;_{ik}\sqrt{2\pi&space;}}e^{\frac{-\left&space;(&space;x-\mu&space;_{ik}&space;\right&space;)^{2}}{2\sigma&space;_{ik}^{2}}}" title="P\left ( X_{i}=x|Y=y_{k} \right )=\frac{1}{\sigma _{ik}\sqrt{2\pi }}e^{\frac{-\left ( x-\mu _{ik} \right )^{2}}{2\sigma _{ik}^{2}}}" /></a>


```python
'''
    return (integer) the (0-based) index of class to which the document belongs
    
    arguments:
    num_classes: (int) number of classes
    num_features: (int) number of features
    prior: (1-d array) of dim num_classes
           (prior probability of a set of features belonging to a class)
    mean_std: (3-d array) of dim num_classes x num_features x 2 (2: mean and std)
              (mean and standard deviation for all features, given the class)
    x: (list) of features
'''
def apply_gaussian_naive_bayes(num_classes, num_features, prior, mean_std, x):
    #initialize score for each class to zero 
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

### Learn

#### Divide data into Train and Test


```python
#mask a % of data for training, and the remaining for testing
mask = rand(len(df)) < train_ds_percent
df_train = df[mask]
df_test = df[~mask]
```

#### Learn


```python
#train the prior and likelihood on observed data df_train
classes, features, prior, mean_std = train_gaussian_nb(df_train, class_colname)
```

## Prediction

### Predict


```python
#iterate over test dataset and count the number of correct and incorrect predictions
count_correct, count_incorrect = 0, 0
for index, row in df_test.iterrows():
    #actual class
    actual_cls = row[class_colname]
    #predicted class
    # input provided as row[:-1].to_list(), means, all columns except last, converted to a list
    pred_cls = apply_gaussian_naive_bayes(len(classes), len(features), prior, mean_std, row[:-1].to_list())
    if classes[pred_cls] == actual_cls:
        count_correct += 1
    else:
        count_incorrect += 1
    #print('(predicted, actual):', classes[pred_cls], row[class_colname])

```

### <font color=magenta>Prediction Accuracy</font>


```python
print('Correct: ', count_correct, 'Incorrect: ', count_incorrect)
print('Percentage of correct predictions: ', (count_correct * 100)/(count_correct + count_incorrect))
```

    Correct:  36 Incorrect:  2
    Percentage of correct predictions:  94.73684210526316
    

## Sidebar - Comparison of Naive Bayes and Logistic Regression

> [Generative and Discriminative Classifiers : Naive Bayes and Logistic Regression - <font color=magenta>Tom Mitchell</font>](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf)

### <font color=magenta>Same Parametric Form</font> (under attribute independence assumptions)

The parametric form of P(Y\|X) used by Logistic Regression is precisely the form implied by the assumptions of a Gaussian Naive Bayes clasifier.

<img src="{{site.baseurl}}/assets/images/iris-gnb/relation_gaussian_nb_to_logreg.png">

### <font color=magenta>Attribute Independence Assumption</font>; <font color=magenta>Convergence</font>; <font color=magenta>Input Data Size</font>

<img src="{{site.baseurl}}/assets/images/iris-gnb/relation_gaussian_nb_to_logreg_indep_assump_and_convergence.png">

### <font color=magenta>Asymptotic Comparison</font>

<img src="{{site.baseurl}}/assets/images/iris-gnb/gnb_logreg_asympto_comp.png">

### <font color=magenta>Summary</font>

<img src="{{site.baseurl}}/assets/images/iris-gnb/gnb_logreg_comp_summary.png">
