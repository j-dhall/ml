![png]({{site.baseurl}}/assets/images/output_46_0.png)
![jpg]({{site.baseurl}}/assets/images/query.jpg)

### Imports


```python
import re #data cleaning
import string #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing\
import TextVectorization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.initializers import Constant


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
```

### Environment


```python
data_file_train = '..\\input\\movie-review-sentiment-analysis-kernels-only\\train.tsv'
data_file_test = '..\\input\\movie-review-sentiment-analysis-kernels-only\\test.tsv'
embed_file_glove = '..\\input\\glove-global-vectors-for-word-representation\\glove.6B.200d.txt'
embed_file_word2vec = '..\\input\\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin'
```


```python
embed_dim = 300 #this has to match the embed_file we choose above
```

### Data

#### read the data files into data frames


```python
pd.set_option('display.max_colwidth', -1)
df_train = pd.read_csv(data_file_train, delimiter='\t')
df_test = pd.read_csv(data_file_test, delimiter='\t') #for submission
print('***Training Set:***\n', df_train.head(1))
print('***Testing Set:***\n', df_test.head(1))
```

#### cleaning the data


```python
#drop unwanted columns
def retain_cols(df, cols):
    df = df[cols]
    return df

#Turn url's into url, remove anything that's not alphanumeric or a space.
#Then lowercase what's left.
def clean_str(in_str):
    in_str = str(in_str)
    # replace urls with 'url'
    in_str = re.sub(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\
    [a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\
    [a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}\
    |https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\
    \.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})",\
                    "url", in_str)
    in_str = re.sub(r'([^\s\w]|_)+', '', in_str)
    return in_str.strip().lower()

#clean the data
def clean_data(df, cols):
    df = retain_cols(df, cols)
    df['Text'] = df['Phrase'].apply(clean_str)
    return df

df_train = clean_data(df_train, ['Phrase', 'Sentiment'])
df_test = clean_data(df_test, ['Phrase']) #for submission
```

#### balancing classes

Our data is classified into 5 different classes, very negative, slightly negative, neutral, slightly positive and very positive.

Sadly our dataset isn't balanced, so we need to do that ourselves

NOTE: <font color='orange'>using alias 'df' for 'df_train'</font> for convenience


```python
df = df_train
```


```python
df.Sentiment.value_counts()
```


```python
df_0 = df[df['Sentiment'] == 0].sample(frac=1)
df_1 = df[df['Sentiment'] == 1].sample(frac=1)
df_2 = df[df['Sentiment'] == 2].sample(frac=1)
df_3 = df[df['Sentiment'] == 3].sample(frac=1)
df_4 = df[df['Sentiment'] == 4].sample(frac=1)
```


```python
# we want a balanced set for training against - there are 7072 `0` examples
sample_size = 7072

data = pd.concat([df_0.head(sample_size),\
                  df_1.head(sample_size),\
                  df_2.head(sample_size),\
                  df_3.head(sample_size),\
                  df_4.head(sample_size)]).sample(frac=1)
data.head(3)
```

NOTE: <font color='orange'>'data' represents 'df_train' after data preprocessing</font>

#### input vectorization

##### sequence length and max features


```python
data['l'] = data['Text'].apply(lambda x: len(str(x).split(' ')))
print("mean length of sentence: " + str(data.l.mean()))
print("max length of sentence: " + str(data.l.max()))
print("std dev length of sentence: " + str(data.l.std()))
```


```python
# these sentences aren't that long so we may as well use the whole string
sequence_length = int(data.l.max())
```


```python
max_features = 20000 # this is the number of words we care about
```

##### vocabulary


```python
vectorizer = TextVectorization(max_tokens=max_features,\
                               output_sequence_length=sequence_length)
text_ds = tf.data.Dataset.from_tensor_slices(data['Text'].values).batch(128)
vectorizer.adapt(text_ds)
voc = vectorizer.get_vocabulary()
print('Vocabulary:', voc[:5])
print('Vectorizing "the cat sat on the mat":\n',\
     vectorizer([["the cat sat on the mat"]]).numpy()[0, :6])
word_index = dict(zip([x.decode('utf-8') for x in voc], range(len(voc))))
print('Word index of words ["the", "cat", "sat", "on", "the", "mat"]:', \
      [word_index[w] for w in ["the", "cat", "sat", "on", "the"]])#, "mat"
#'mat' is not in the vocabulary.
#hence, vocabulary index is 1 (index 0 for empty token)
#(index 2 onwards for words in vocabulary)
```

NOTE: The index of vectorizing is off by 2 when compared with word index generated from its vocabulary. We will have to <font color='blue'>make adjustment of 2 when creating the embedding matrix.</font>

##### vectorize input and split for validation


```python
X = vectorizer(np.array([[s] for s in data['Text'].values])).numpy()
y = np.array(pd.get_dummies(data['Sentiment']).values)
```


```python
# where there isn't a test set, Kim keeps back 10% of the data for testing,
#I'm going to do the same since we have an ok amount to play with
X_train, X_val, y_train, y_val =\
train_test_split(X, y, test_size=0.1)
print("Validation set size " + str(len(X_val)))
```

### Word Embeddings

#### load GloVe embeddings

embeddings_index = {}
f_embed = open(embed_file_glove, encoding="utf8")
for line in f_embed:
    tokens = line.split()
    word = tokens[0]
    coefs = np.asarray(tokens[1:], dtype='float32')
    embeddings_index[word] = coefs
f_embed.close()
print('Found %s word vectors' % len(embeddings_index))

#### load word2vec embeddings


```python
from gensim.models import KeyedVectors
word2vec = KeyedVectors.load_word2vec_format(embed_file_word2vec, binary=True)
```


```python
embeddings_index = word2vec
```

#### embedding matrix


```python
num_words = min(max_features, len(word_index)) + 2
embedding_dim = embed_dim
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    try:
        embedding_vector = embeddings_index[word]
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i+2] = embedding_vector
            hits += 1
        else:#for non-exception throwing embeddings_index
            misses += 1
    except KeyError: #for non-exception throwing embeddings_index
        misses += 1
        continue

print("Converted %d words (%d misses)" % (hits, misses))
```

### Model - CNN for Sentence Classification - Yoon Kim

https://www.kaggle.com/hamishdickson/cnn-for-sentence-classification-by-yoon-kim

#### model architecture


```python
num_filters = 100
```


```python
inputs_3 = Input(shape=(sequence_length,), dtype='int32')
embedding_layer_3 = Embedding(num_words,
                            embedding_dim,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=sequence_length,
                            trainable=True)(inputs_3)

reshape_3 = Reshape((sequence_length, embedding_dim, 1))(embedding_layer_3)

# note the relu activation
conv_0_3 = Conv2D(num_filters, kernel_size=(3, embedding_dim), activation='relu', kernel_regularizer=regularizers.l2(3))(reshape_3)
conv_1_3 = Conv2D(num_filters, kernel_size=(4, embedding_dim), activation='relu', kernel_regularizer=regularizers.l2(3))(reshape_3)
conv_2_3 = Conv2D(num_filters, kernel_size=(5, embedding_dim), activation='relu', kernel_regularizer=regularizers.l2(3))(reshape_3)

maxpool_0_3 = MaxPool2D(pool_size=(sequence_length - 3 + 1, 1), strides=(1,1), padding='valid')(conv_0_3)
maxpool_1_3 = MaxPool2D(pool_size=(sequence_length - 4 + 1, 1), strides=(1,1), padding='valid')(conv_1_3)
maxpool_2_3 = MaxPool2D(pool_size=(sequence_length - 5 + 1, 1), strides=(1,1), padding='valid')(conv_2_3)

concatenated_tensor_3 = Concatenate(axis=1)([maxpool_0_3, maxpool_1_3, maxpool_2_3])
flatten_3 = Flatten()(concatenated_tensor_3)

dropout_3 = Dropout(0.5)(flatten_3)
output_3 = Dense(units=5, activation='softmax')(dropout_3)
```


```python
model_3 = Model(inputs=inputs_3, outputs=output_3)
model_3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_3.summary())
```

#### fit the model


```python
batch_size = 32
history_3 = model_3.fit(X_train, y_train,\
                        epochs=20, batch_size=batch_size,\
                        verbose=1, validation_split=0.2)
```

#### plot training and validation accuracy


```python
plt.plot(history_3.history['accuracy'])
plt.plot(history_3.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history_3.history['loss'])
plt.plot(history_3.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
```

#### accuracy


```python
y_hat_3 = model_3.predict(X_val)
accuracy_score(list(map(lambda x: np.argmax(x), y_val)),\
               list(map(lambda x: np.argmax(x), y_hat_3)))
```

#### confusion matrix


```python
confusion_matrix(list(map(lambda x: np.argmax(x), y_val)),\
                 list(map(lambda x: np.argmax(x), y_hat_3)))
```

### Testing

#### sanity check


```python
X_test = vectorizer(np.array([[s] for s in ['amazing', 'thrilling experience', 'expected more', 'it was fun', 'waste of time']])).numpy()
y_test = model_3.predict(X_test)
[(i+1, list(y).index(max(y))) for i, y in enumerate(y_test)]
```

#### submission


```python
X_test = vectorizer(np.array([[s] for s in df_test['Text'].values])).numpy()
y_test = model_3.predict(X_test)
y_test_class = [list(y).index(max(y)) for i, y in enumerate(y_test)]
```


```python
data_file_sub = '..\\input\\movie-review-sentiment-analysis-kernels-only\\sampleSubmission.csv'
df_sub = pd.read_csv(data_file_sub)
df_sub['Sentiment'] = np.asarray(y_test_class)
df_sub.to_csv("cnn.csv", index=False)
```

### Model - Bidirectional GRU and LSTM

https://www.kaggle.com/artgor/movie-review-sentiment-analysis-eda-and-models

#### Imports


```python
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, \
Callback, EarlyStopping
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout,\
Activation, Conv1D, GRU, GRU, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
```

#### Constants


```python
max_len = sequence_length
embed_size = embedding_dim
```


```python
embed_size
```


```python
def build_model1(lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, kernel_size1=3, kernel_size2=2, dense_units=128, dr=0.1, conv_size=32):
    file_path = "best_model.hdf5"
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)
    
    inp = Input(shape = (max_len,))
    x = Embedding(num_words, embed_size, weights = [embedding_matrix], trainable = False)(inp)
    x1 = SpatialDropout1D(spatial_dr)(x)

    x_gru = Bidirectional(GRU(units, return_sequences = True))(x1)
    x1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool1_gru = GlobalAveragePooling1D()(x1)
    max_pool1_gru = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool3_gru = GlobalAveragePooling1D()(x3)
    max_pool3_gru = GlobalMaxPooling1D()(x3)
    
    x_lstm = Bidirectional(LSTM(units, return_sequences = True))(x1)
    x1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool1_lstm = GlobalAveragePooling1D()(x1)
    max_pool1_lstm = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool3_lstm = GlobalAveragePooling1D()(x3)
    max_pool3_lstm = GlobalMaxPooling1D()(x3)
    
    
    x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool3_gru, max_pool3_gru,
                    avg_pool1_lstm, max_pool1_lstm, avg_pool3_lstm, max_pool3_lstm])
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(dense_units, activation='relu') (x))
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(int(dense_units / 2), activation='relu') (x))
    x = Dense(5, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    history = model.fit(X_train, y_train, batch_size = 128, epochs = 2, validation_split=0.1, 
                        verbose = 1, callbacks = [check_point, early_stop])
    model = load_model(file_path)
    return model, history
```


```python
model1, history1 = build_model1(lr = 1e-3, lr_d = 1e-10, units = 64,\
                      spatial_dr = 0.3, kernel_size1=3, kernel_size2=2,\
                      dense_units=32, dr=0.1, conv_size=32)
```


```python
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
```

#### accuracy


```python
y_hat_3 = model1.predict(X_val)
accuracy_score(list(map(lambda x: np.argmax(x), y_val)),\
               list(map(lambda x: np.argmax(x), y_hat_3)))
```

#### confusion matrix


```python
confusion_matrix(list(map(lambda x: np.argmax(x), y_val)),\
                 list(map(lambda x: np.argmax(x), y_hat_3)))
```

### Testing

#### sanity check


```python
X_test = vectorizer(np.array([[s] for s in ['amazing', 'thrilling experience', 'expected more', 'it was fun', 'waste of time']])).numpy()
y_test = model1.predict(X_test)
[(i+1, list(y).index(max(y))) for i, y in enumerate(y_test)]
```

#### submission


```python
X_test = vectorizer(np.array([[s] for s in df_test['Text'].values])).numpy()
y_test = model1.predict(X_test)
y_test_class = [list(y).index(max(y)) for i, y in enumerate(y_test)]
```


```python
data_file_sub = '..\\input\\movie-review-sentiment-analysis-kernels-only\\sampleSubmission.csv'
df_sub = pd.read_csv(data_file_sub)
df_sub['Sentiment'] = np.asarray(y_test_class)
df_sub.to_csv("bigru_lstm.csv", index=False)
```
