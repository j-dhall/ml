---
layout: post
title: "[without library] Text Classification using Multinomial Naive Bayes"
date: 2021-02-01 16:30:59 +0530
categories:
---


## <font color=orange>Multinomial Naive Bayes for Text Classification</font>
#### Introduction to Information Retrieval - Manning
url: https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html

### Imports


```python
import pandas as pd #input
import numpy as np #counts
#vocabulary, count of tokens
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from math import log # to calculate posterior probability
```

### Constants

It is good to have max_tokens higher than vocab_size = len(vocabulary), so that we can make use to all the words in the vocabulary for prediction. Otherwise, only the first max_tokens words of the vocabulary will be used.


```python
max_tokens = 1000
```

### Environment

#### Naive Bayes dataset - Introduction to Information Retrieval - Manning
url: https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html

<img src="{{site.baseurl}}/assets/images/Naive Bayes dataset - Introduction to Information Retrieval - Manning.png">


```python
f_train = '../input/naive_bayes_demo_data_china/train.csv'
f_test = '../input/naive_bayes_demo_data_china/test.csv'
f_classes = '../input/naive_bayes_demo_data_china/classes.csv'
cols_train = ['class', 'title', 'body']
cols_test = ['class', 'title', 'body']
cols_classes = ['class_name']
```

#### AG News

url: https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv


```python
f_train = '../input/ag_news_csv/train.csv'
f_test = '../input/ag_news_csv/test.csv'
f_classes = '../input/ag_news_csv/classes.txt'
cols_train = ['class', 'title', 'body']
cols_test = ['class', 'title', 'body']
cols_classes = ['class_name']
```

### Data


```python
df_train = pd.read_csv(f_train,
                       header=None)
df_train.columns = cols_train
```


```python
df_test = pd.read_csv(f_test,
                      header=None)
df_test.columns = cols_test
```


```python
df_classes = pd.read_csv(f_classes,
                         header=None)
df_classes.columns = cols_classes
```

#### Sanity check for data getting loaded


```python
print(df_train.head(2))
print(df_test.head(2))
print(df_classes.head(2))
```

### Helper Functions


```python
"""
    return (vectorizer, list) the vocabulary of words that occur in the documents
    
    arguments:
    vec: vectorizer (of type tensorflow.keras.layers.experimental.preprocessing.TextVectorization)
    df_docs: dataframe with documents
    df_docs_colname: column name of the column having the documents
"""
def extract_vocabulary(vec, df_docs, doc_colname='body'):
    #WARNING: https://stackoverflow.com/questions/62679020/error-while-trying-to-save-npy-numpy-file
    #VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
    vec.adapt(df_docs[doc_colname].values)
    return vec, vec.get_vocabulary()



"""
    return (number) the number of rows in the dataframe, each row corresponding to a document
    
    arguments:
    df_docs: dataframe with documents (can have columns in addition to the document)
"""
def count_docs(df_docs):
    return len(df_docs)



"""
    return (number) the number of rows in the dataframe belonging to class 'c'
    
    arguments:
    df_docs: dataframe with documents (should have a 'class' column in addition to the document)
    c: the class for which the document count is sought
    class_colname: column name of the column having class name/id
"""
def count_docs_in_class(df_docs, c, class_colname='class'):
    return count_docs(df_docs[df_docs[class_colname] == c])



"""
    return (string) the concatenated text of all documents of class c
    
    arguments:
    df_docs: dataframe with documents (should have a 'class' column in addition to the document)
    c: the class for which the documents are sought
    df_docs_colname: column name of the column having the documents
    class_colname: column name of the column having class name/id
"""
def concatenate_text_of_all_docs_in_class(df_docs, c, doc_colname='body', class_colname='class'):
    #for df[class == c] get the 'body' column, convert its type to list,
    # and concatenate all lists, separated by space.
    return ' '.join(df_docs[df_docs[class_colname] == c][doc_colname].to_list())



'''
    return (numpy array) an array of counts of tokens in the text,
    each array element corresponding to the count of token at that index in the vocabulary.
    
    arguments:
    vec: output_mode="count" based vectorizer
        that was 'adapt'ed to the documents to learn the vocabulary.
    text: concatenated documents belonging to a certain class
'''
def count_tokens_of_all_terms(vec, text):
    counts = vec([[text]])
    return counts[0].numpy()
```

### Model

#### Training Algorithm

<img src="{{site.baseurl}}/assets/images/TrainMultinomialNB.png">


```python
'''
    return 
            vectorizer,
            
            prior: 1-d array of dim num_classes
            (prior probability of document belonging to class),
            
            cond_prob: 2-d array of dim token x num_classes
            (conditional probability for all token, given the class)
            
    arguments:
    df_classes: dataframe with class names
    df_docs: dataframe with documents (should have a 'class' column in addition to the document 'body' column).
                provide suitable column names otherwise, using the doc_colname and class_colname arguments.
'''
def train_multinomial_nb(df_classes, df_docs,
                         doc_colname='body', class_colname='class'):
    
    #extract vocabulary
    
    #create an instance of TextVectorization
    vectorizer = TextVectorization(max_tokens=max_tokens, output_mode="count")
    vectorizer, vocabulary = extract_vocabulary(vectorizer, df_docs, doc_colname)
    
    #data structures for prior and conditional probabilities
    
    num_classes = len(df_classes) #number of classes of documents
    prior = np.zeros((num_classes), dtype=float) #prior probability of document belonging to a class
    vocab_size = len(vocabulary)
    cond_prob = np.zeros((vocab_size, num_classes), dtype=float)

    #total number of documents, to calculate prior probability of document belonging to a class
    N = count_docs(df_docs)
    
    #for each class...
    for c in range(num_classes):
        cls = c+1 #c is 0-based index, but class is 1-based index

        #calculate prior probability of document belonging to class cls
        N_c = count_docs_in_class(df_docs, cls, class_colname) #number of document of class cls
        prior[c] = N_c / N # (num_docs of class cls / total num_docs)

        #calculate likelihood: conditional probability for all token, given this class 

        #1. concatenate documents of this class
        text_c = concatenate_text_of_all_docs_in_class(df_docs, cls, doc_colname, class_colname)
        #2. in this concatenated text, get counts for each token
        T_ct_all_t = count_tokens_of_all_terms(vectorizer, text_c)
        #3. calculate denominator of conditional probability: sigma(T_ct) + |V|
        den = T_ct_all_t.sum() + vocab_size
        #4 set the likelihood: conditional probability for all tokens, given this class
        for t in range(vocab_size):
            t_idx = t+1 #vectorizer begins with an OOV, so, add 1 to accommodate OOV
            cond_prob[t][c] = (T_ct_all_t[t_idx] + 1)/den
            
    return vectorizer, prior, cond_prob
```

#### Prediction Algorithm

<img src="{{site.baseurl}}/assets/images/ApplyMultinomialNB.png">


```python
'''
    return (integer) the (0-based) index of class to which the document belongs
    
    arguments:
    df_classes: dataframe with class names
    vectorizer
    prior: 1-d array of dim num_classes (prior probability of document belonging to class)
    cond_prob: 2-d array of dim token x num_classes (conditional probability for all token, given the class)
    text: document whose class is to be predicted
'''
def apply_multinomial_nb(df_classes, vectorizer, prior, cond_prob, text):
    #count of each token
    W = count_tokens_of_all_terms(vectorizer, text)

    #vocabulary
    vocabulary = vectorizer.get_vocabulary()
    vocab_size = len(vocabulary)

    num_classes = len(df_classes) #number of classes of documents
    score = np.zeros((num_classes), dtype=float)

    #for each class, calculate the posterior probability

    #for each class...
    for c in range(num_classes):

        #for this class, add the log-prior probability to the score
        score[c] += log(prior[c], 10) #log to the base 10

        #for each term, add the log-likelihood to the score

        for t in range(vocab_size):
            t_idx = t+1 #vectorizer begins with an OOV, so, add 1 to accommodate OOV
            T_ct = W[t_idx] #count of number of times this token appeared in text

            #for a term appearing multiple times, repeatedly add the log-likelihood to the score
            for j in range(T_ct):
                score[c] += log(cond_prob[t][c], 10) #log to the base 10
                
    return score.argmax()
```

#### Training


```python
#train the model
vectorizer, prior, cond_prob = train_multinomial_nb(df_classes, df_train)
print('vocabulary length is:', len(vectorizer.get_vocabulary()))
print('vocabulary is:', vectorizer.get_vocabulary()[:10])
```

#### Prediction


```python
#predict the documents
N = len(df_test)
count_correct, count_incorrect = 0, 0
for i in range(1000):
    text = df_test['body'][i] #input document for classification
    pred_cls = apply_multinomial_nb(df_classes, vectorizer, prior, cond_prob, text)
    actual_cls = df_test['class'][i] - 1 #df_test.class is 1-based index, but df_classes and pred_class are 0-based index
    #print('(predicted, actual):', df_classes['class_name'][pred_cls], df_classes['class_name'][actual_cls])
    if pred_cls == actual_cls:
        count_correct += 1
    else:
        count_incorrect += 1
print('Correct: ', count_correct, 'Incorrect: ', count_incorrect)
print('Percentage of correct predictions: ', (count_correct * 100)/(count_correct + count_incorrect))
```
