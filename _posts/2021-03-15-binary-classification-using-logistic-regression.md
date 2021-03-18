---
layout: post
title: "[without library] Binary Classification using Logistic Regression"
date: 2021-02-18 19:57:59 +0530
categories:
---

# [without library] <font color="orange">Binary Classification using Logistic Regression
</font>

### Based on - [2010] Generative and Discriminative Classifiers : Naive Bayes and Logistic Regression - <font color=magenta>Tom Mitchell</font>

<img src="{{site.baseurl}}/assets/images/my-notes-logistic-regression-images/log_reg_display.png">

## Introduction

This notebook implements <font color=blue>Binomial Logistic Regression</font>. It performs <font color=blue>binary classification</font> on a generated dataset consisting of two gaussian distributed clusters of points in a 2-dimensional space. The prediction accuracy for a learning dataset of 100 points is <font color=blue>90+%</font> if the clusters overlap slightly and <font color=blue>98-100%</font> if the clusters do not overlap.

Resources:
- [Generative and Discriminative Classifiers : Naive Bayes and Logistic Regression](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf)

Towards the end, I have included a <font color=blue>sidebar</font> on the <font color=magenta>Binomial Logistic Regression</font>. 

## Taxonomy and Notes

Logistic Regression is a <font color=blue>probabilistic</font> classifier. But, it does not model the complete distribution P(X, Y). It is only interested in discriminating among classes. It does that by __computing P(Y\|X) directly__ from the training data. Hence, it is a <font color=blue>probabilistic-discriminative</font> classifier. The classifier's <font color=magenta>sigmoid</font> function is <font color=blue>linear</font> in terms of weights and bias for the features. It is <font color=blue>parametric</font> (weights and bias are the parameters). Binomial Logistic Regression is <font color=blue>binary</font>. The model can be modified to use <font color=magenta>softmax</font> instead of sigmoid, and it becomes <font color=blue>Multiclass</font> Logistic Regression.

In summary, Logistic Regression is:
- Probabilistic
- Discriminative
- Binary and Multiclass
- Linear
- Parametric

The density estimation of P(Y\|X) is parametric, <font color=blue>point estimation</font>, using <font color=blue>Maximum Likelihood Estimation (MLE)</font>.

The computation of P(Y\|X) is <font color=red>not in closed-form</font>. So, a numerical method like <font color=blue>Gradient Descent</font> is used to optimize the model's cost function.

<img src="{{site.baseurl}}/assets/images/my-notes-logistic-regression-images/log_reg_mmap.png">

## Imports


```python
from math import log # to calculate posterior probability
import numpy as np #arrays for data points
from numpy.random import rand, normal, randint #gaussian distributed data points
from numpy import dot #vector dot product for the linear kernel
from numpy import mean, std #mean and standard deviation for gaussian probabilities
from scipy.stats import norm #gaussian probabilities
import pandas as pd #input
import seaborn as sns #plotting
import matplotlib.pyplot as plt #plotting
%matplotlib inline
```

## X_m, Y

### Data Configuration


```python
M = 100 #number of data points
cols = ['X0', 'X1', 'X2', 'Y'] #column names of the dataframe
n_features = len(cols)-1 #number of dimensions
K = 2 #number of classes
loc_scale = [(5, 1), (7, 1)] #mean and std of data points belonging to each class
```

### Generate Data
Gaussian clusters in 2D numpy arrays


```python
def generate_X_m_and_Y(M, K, n_features, loc_scale):
    #X_m, Y
    # we use this extra count (+1) to accomodate for X0 = 1 (the attribute for bias)
    X_m = np.ones((K, (int)(M/2), n_features), dtype=float) #initialize data points
    Y = np.empty((K, (int)(M/2)), dtype=int) #initialize the class labels

    for k in range(K): #for each class, generate data points #create data points for each class
        #create data points for class k using gaussian (normal) distribution
        X_m[k][:, 1:] = normal(loc=loc_scale[k][0], scale=loc_scale[k][1], size=((int)(M/2), n_features-1))
        #append features/columns after the bias (first) column
        #X_m[:, 1:] = X
        #create labels (0, 1) for class k (0, 1).
        Y[k] = np.full(((int)(M/2)), k, dtype=int)
    X_m = X_m.reshape(M, n_features) #collapse the class axis
    Y = Y.reshape(M) #collapse the class axis
    X_m.shape, Y.shape #print shapes
    
    return X_m, Y

X_m, Y = generate_X_m_and_Y(M, K, n_features, loc_scale)
X_m.shape, Y.shape
```




    ((100, 3), (100,))



### X_m, Y in DataFrame


```python
def create_df_from_array(X_m, Y, n_features, cols):
    #create series from each column of X_m, and a series from Y
    l_series = [] #list of series, one for each column
    for feat in range(n_features): #create series from each column of X_m
        l_series.append(pd.Series(X_m[:, feat])) #create series from a column of X_m
    l_series.append(pd.Series(Y[:])) #create series from Y

    frame = {col : series for col, series in zip(cols, l_series)} #map of column names to series
    df = pd.DataFrame(frame) #create dataframe from map

    return df

df = create_df_from_array(X_m, Y, n_features, cols)
df.sample(n = 10)
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
      <th>X0</th>
      <th>X1</th>
      <th>X2</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>89</th>
      <td>1.0</td>
      <td>5.582885</td>
      <td>7.660471</td>
      <td>1</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1.0</td>
      <td>4.320468</td>
      <td>5.115238</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>5.034319</td>
      <td>5.274385</td>
      <td>0</td>
    </tr>
    <tr>
      <th>91</th>
      <td>1.0</td>
      <td>5.784753</td>
      <td>7.475163</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1.0</td>
      <td>6.418753</td>
      <td>3.360426</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>1.0</td>
      <td>5.006697</td>
      <td>3.317916</td>
      <td>0</td>
    </tr>
    <tr>
      <th>93</th>
      <td>1.0</td>
      <td>6.344121</td>
      <td>7.284477</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.0</td>
      <td>5.568940</td>
      <td>2.963496</td>
      <td>0</td>
    </tr>
    <tr>
      <th>85</th>
      <td>1.0</td>
      <td>6.859079</td>
      <td>5.874983</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1.0</td>
      <td>5.417719</td>
      <td>5.529503</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#scatter plot of data points
#class column Y is passed in as hue
sns.scatterplot(x=cols[1], y=cols[2], hue=cols[3], data=df)
```




    <AxesSubplot:xlabel='X1', ylabel='X2'>




    
![png]({{site.baseurl}}/assets/images/logreg_bino_output_19_1.png)
    


## Model

### Model Configuration


```python
learning_rate = 0.001
convergence_cost_diff = 0.0005
```

### Linear Combination of Weights / Coefficients and Features

<a href="https://www.codecogs.com/eqnedit.php?latex=w_{0}&plus;\sum_{i=1}^{n}w_{i}X_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_{0}&plus;\sum_{i=1}^{n}w_{i}X_{i}" title="w_{0}+\sum_{i=1}^{n}w_{i}X_{i}" /></a>


```python
def lin_com(W, X):
    return np.dot(W, X)
```

### Probability of -ve class

<a href="https://www.codecogs.com/eqnedit.php?latex=P\left&space;(&space;Y=0|X&space;\right&space;)=\frac{1}{1&plus;e^{\left&space;(w_{0}&plus;\sum_{i=1}^{n}w_{i}X_{i}&space;\right&space;)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P\left&space;(&space;Y=0|X&space;\right&space;)=\frac{1}{1&plus;e^{\left&space;(w_{0}&plus;\sum_{i=1}^{n}w_{i}X_{i}&space;\right&space;)}}" title="P\left ( Y=0|X \right )=\frac{1}{1+e^{\left (w_{0}+\sum_{i=1}^{n}w_{i}X_{i} \right )}}" /></a>


```python
def prob_y0_x(w, x):
    lc = lin_com(w, x)
    return 1/(1 + np.exp(lc))
```

### Probability of +ve class

<a href="https://www.codecogs.com/eqnedit.php?latex=P\left&space;(&space;Y=1|X&space;\right&space;)=\frac{e^{\left&space;(&space;w_{0}&plus;\sum_{i=1}^{n}w_{i}X_{i}&space;\right&space;)}}{1&plus;e^{\left&space;(w_{0}&plus;\sum_{i=1}^{n}w_{i}X_{i}&space;\right&space;)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P\left&space;(&space;Y=1|X&space;\right&space;)=\frac{e^{\left&space;(&space;w_{0}&plus;\sum_{i=1}^{n}w_{i}X_{i}&space;\right&space;)}}{1&plus;e^{\left&space;(w_{0}&plus;\sum_{i=1}^{n}w_{i}X_{i}&space;\right&space;)}}" title="P\left ( Y=1|X \right )=\frac{e^{\left ( w_{0}+\sum_{i=1}^{n}w_{i}X_{i} \right )}}{1+e^{\left (w_{0}+\sum_{i=1}^{n}w_{i}X_{i} \right )}}" /></a>


```python
def prob_y1_x(w, x):
    lc = lin_com(w, x)
    return np.exp(lc)/(1 + np.exp(lc))
```

### <font color='magenta'>Conditional Data Log-Likelihood</font>

<img src="{{site.baseurl}}/assets/images/cross-entropy.png">

If we look at the equation below, the term 'Y ln P(Y)' is the <font color=blue>cross entropy between the true probability Y (=1) and the predicted probability</font>. Since when Y=1, we have (1-Y) = 0, and when Y=0, we have (1-Y) = 1, only one term per data point is non-zero. So, conditional data log-likelihood is basically a <font color=magenta>sum of cross entropies</font>. The lower the cross entropy, the better the prediction. Hence, this function is a cost (loss) function.

<a href="https://www.codecogs.com/eqnedit.php?latex=l(W)=\sum_{l}^{}Y^{l}ln\;&space;P\left&space;(Y^{l}=1|X^{l},W&space;\right&space;)&plus;\left&space;(1-Y^{l}&space;\right&space;)ln\;&space;P\left&space;(Y^{l}=0|X^{l},W&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l(W)=\sum_{l}^{}Y^{l}ln\;&space;P\left&space;(Y^{l}=1|X^{l},W&space;\right&space;)&plus;\left&space;(1-Y^{l}&space;\right&space;)ln\;&space;P\left&space;(Y^{l}=0|X^{l},W&space;\right&space;)" title="l(W)=\sum_{l}^{}Y^{l}ln\; P\left (Y^{l}=1|X^{l},W \right )+\left (1-Y^{l} \right )ln\; P\left (Y^{l}=0|X^{l},W \right )" /></a>


```python
#conditional data log-likelihood ln(P(Y|X,W))
def cond_data_log_likelihood(X_m, Y, w):
    likelihood = 0.0
    for i in range(len(X_m)):
        likelihood += (Y[i]*log(prob_y1_x(w, X_m[i])) + (1 - Y[i])*log(prob_y0_x(w, X_m[i])) )
    return (likelihood)
```

### <font color='magenta'>Gradient along attribute 'i'</font>

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;l\left&space;(&space;W&space;\right&space;)}{\partial&space;w_{i}}=\sum_{l}^{}X_{i}^{l}\left&space;(&space;Y^{l}-\hat{P}\left&space;(Y^{l}=1|X^{l},W&space;\right&space;)&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;l\left&space;(&space;W&space;\right&space;)}{\partial&space;w_{i}}=\sum_{l}^{}X_{i}^{l}\left&space;(&space;Y^{l}-\hat{P}\left&space;(Y^{l}=1|X^{l},W&space;\right&space;)&space;\right&space;)" title="\frac{\partial l\left ( W \right )}{\partial w_{i}}=\sum_{l}^{}X_{i}^{l}\left ( Y^{l}-\hat{P}\left (Y^{l}=1|X^{l},W \right ) \right )" /></a>


```python
#gradient along the attribute 'j'
def gradient(X_m, Y, W, j):
    grad = 0.0
    #iterate over all data-points
    for i in range(len(X_m)):
        grad += X_m[i][j]*(Y[i] - prob_y1_x(W, X_m[i]))
    return grad
```

### <font color='magenta'>Gradients along attributes</font>


```python
#gradient along each attribute
def gradients(X_m, Y, W):
    #gradient along each attribute
    grads = np.zeros(len(W), dtype=float)
    for j in range(len(W)):
        grads[j] = gradient(X_m, Y, W, j)
        
    return grads
```

### Apply gradients on coefficients

<a href="https://www.codecogs.com/eqnedit.php?latex=w_{i}\leftarrow&space;w_{i}&plus;\eta&space;\sum_{l}^{}X_{i}^{l}\left&space;(&space;Y^{l}-\hat{P}\left&space;(Y^{l}=1|X^{l},W&space;\right&space;)&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_{i}\leftarrow&space;w_{i}&plus;\eta&space;\sum_{l}^{}X_{i}^{l}\left&space;(&space;Y^{l}-\hat{P}\left&space;(Y^{l}=1|X^{l},W&space;\right&space;)&space;\right&space;)" title="w_{i}\leftarrow w_{i}+\eta \sum_{l}^{}X_{i}^{l}\left ( Y^{l}-\hat{P}\left (Y^{l}=1|X^{l},W \right ) \right )" /></a>


```python
def apply_gradient(W, grads, learning_rate):
    return (W + (learning_rate * grads))
```

### Training Algorithm

<img src="{{site.baseurl}}/assets/images/grad_descent.png">


```python
def train(X_m, Y, W, learning_rate):
    #learn
    prev_max = cond_data_log_likelihood(X_m, Y, W)
    grads = gradients(X_m, Y, W)
    W = apply_gradient(W, grads, learning_rate)
    new_max = cond_data_log_likelihood(X_m, Y, W)
    #summary print
    i_print = 0
    while(abs(prev_max - new_max) > convergence_cost_diff):
        if(i_print % 500) == 0:
            print('Cost:', prev_max)
        prev_max = new_max
        grads = gradients(X_m, Y, W)
        W = apply_gradient(W, grads, learning_rate)
        new_max = cond_data_log_likelihood(X_m, Y, W)
        i_print += 1

    return W
```

### Learn


```python
#weights
W = np.zeros((n_features), dtype=float)
W = train(X_m, Y, W, learning_rate)
```

    Cost: -69.31471805599459
    Cost: -49.707947324850565
    Cost: -40.87854558929088
    Cost: -35.85466773616585
    Cost: -32.65388264792396
    Cost: -30.444903169173813
    Cost: -28.830852125987995
    Cost: -27.600808997259257
    Cost: -26.63289105803656
    Cost: -25.8518897927959
    Cost: -25.20890607913062
    Cost: -24.670769018536692
    Cost: -24.21417569334356
    Cost: -23.822268801602643
    Cost: -23.482546876964133
    Cost: -23.18553887871831
    

#### <font color=magenta>weights (coefficients)</font>


```python
print('The learnt weights (coefficients) are:', W)
```

    The learnt weights (coefficients) are: [-14.97508012   1.16796148   1.35110071]
    

## Prediction

### Generate Data
Gaussian clusters in 2D numpy arrays


```python
X_m, Y = generate_X_m_and_Y(M, K, n_features, loc_scale) #generate test data points
```

### Predict


```python
Y_pred = [prob_y1_x(W, X) for X in X_m]
Y_pred_class = [0 if y < 0.5 else 1 for y in Y_pred] #decision based on predicted margin
```

### Predicted X_m, Y in DataFrame


```python
df = create_df_from_array(X_m, Y, n_features, cols) #create test dataframe
df['Y_pred_margin'] = Y_pred #append the prediction margin column
df['Y_pred_class'] = Y_pred_class #append the class
df.sample(n = 10)
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
      <th>X0</th>
      <th>X1</th>
      <th>X2</th>
      <th>Y</th>
      <th>Y_pred_margin</th>
      <th>Y_pred_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37</th>
      <td>1.0</td>
      <td>5.234051</td>
      <td>5.226969</td>
      <td>0</td>
      <td>0.141882</td>
      <td>0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1.0</td>
      <td>4.146087</td>
      <td>6.144546</td>
      <td>0</td>
      <td>0.138154</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1.0</td>
      <td>6.182099</td>
      <td>3.720378</td>
      <td>0</td>
      <td>0.061340</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1.0</td>
      <td>5.694109</td>
      <td>5.687429</td>
      <td>0</td>
      <td>0.345181</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1.0</td>
      <td>5.386995</td>
      <td>2.323051</td>
      <td>0</td>
      <td>0.003893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1.0</td>
      <td>5.202220</td>
      <td>5.670654</td>
      <td>0</td>
      <td>0.224878</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1.0</td>
      <td>7.099939</td>
      <td>7.113818</td>
      <td>1</td>
      <td>0.949255</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79</th>
      <td>1.0</td>
      <td>7.165353</td>
      <td>6.450322</td>
      <td>1</td>
      <td>0.891757</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1.0</td>
      <td>4.034987</td>
      <td>5.389770</td>
      <td>0</td>
      <td>0.048326</td>
      <td>0</td>
    </tr>
    <tr>
      <th>66</th>
      <td>1.0</td>
      <td>6.995258</td>
      <td>6.390637</td>
      <td>1</td>
      <td>0.861703</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#scatter plot of data points
#class column Y is passed in as hue
sns.scatterplot(x=cols[1], y=cols[2], hue='Y_pred_class', data=df)
```




    <AxesSubplot:xlabel='X1', ylabel='X2'>




    
![png]({{site.baseurl}}/assets/images/logreg_bino_output_59_1.png)
    



```python
Y_pred_np = np.array(Y_pred)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_m[:, 1], X_m[:, 2], Y_pred, c = (Y_pred_np>0.5))
plt.show()
```


    
![png]({{site.baseurl}}/assets/images/logreg_bino_output_60_0.png)
    


### <font color=magenta>Prediction Accuracy</font>


```python
Y_pred_corr = (Y==Y_pred_class)
num_corr = len(Y_pred_corr[Y_pred_corr == True])
print('Accuracy:', (num_corr/M)*100, '%')
```

    Accuracy: 88.0 %
    

## Sidebar - <font color=blue>Binomial Logistic Regression</font>

### P(Y\|X) <font color=magenta>directly</font>?

Actually, the term likelihood can be used for P(X\|Y) as well as P(Y\|X). It can be used for <font color=blue>any distribution</font>. In Naive Bayes, we estimate params (mu/sigma, or frequencies) for distribution P(X\|Y). But, in Logistic Regression, we estimate W (the weight vector W in W.X) to maximize likelihood of the distribution P(Y\|X) -  <font color=blue>this is what we call as estimating P(Y\|X) directly</font>.


<img src="{{site.baseurl}}/assets/images/my-notes-logistic-regression-images/log_reg_mmap_note_likelihood.png">

### <font color=magenta>Linear</font> Boundary

<img src="{{site.baseurl}}/assets/images/my-notes-logistic-regression-images/log_reg_mmap_note_linear.png">

### Why Logistic Regression?

<img src="{{site.baseurl}}/assets/images/my-notes-logistic-regression-images/log_reg_why.png">

### Model and Cost Function

<img src="{{site.baseurl}}/assets/images/my-notes-logistic-regression-images/0001.jpg">

### Derivatives for Gradient Descent

<img src="{{site.baseurl}}/assets/images/my-notes-logistic-regression-images/0002.jpg">

### Regularized Logistic Regression

<img src="{{site.baseurl}}/assets/images/my-notes-logistic-regression-images/0003.jpg">
