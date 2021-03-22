---
layout: post
title: "pandas Playground"
date: 2020-12-16 22:32:59 +0530
categories:
---

# <font color = orange><b>Imports, and Data Fetch</b></font>

**imports**


```python
#imports
import pandas as pd
import numpy as np
```

**read configuration file**


```python
#configuration
from read_config import Config
config = Config ()
```

**data**

NOTE:
We make use of the following dataframes
- **df_titanic** read from the titanic dataset
- **df_raw** - an in-code sample dataframe to demonstrate csv file save, and some dataframe statistics
- **df_books** - an in-code sample dataframe - starting from sub-section 5. of section 'DataFrame Playground'. It is used to show most of the features of the dataframe.

*titanic*


```python
config.set_dataset_id ("titanic")
df_titanic = config.get_train_df ()
df_titanic.head (2)
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



# <font color = orange> DataFrame Playground </font>

## <font color = grey><b>1. Read Data from CSV file into a DataFrame</b></font>


```python
import os
import pandas as pd

config.set_dataset_id ("titanic") #read titanic data
df_titanic = config.get_train_df ()
```

## <font color = grey><b>2. Create DataFrame from Dictionary</b></font>

<font color = magenta>
    pd.DataFrame <b>(dict, columns = , index = )</b>
</font>


```python
raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
        'last_name': ['Miller', 'Jacobson', ".", 'Milner', 'Cooze'], 
        'age': [42, 52, 36, 24, 73], 
        'preTestScore': [4, 24, 31, ".", "."],
        'postTestScore': ["25,000", "94,000", 57, 62, 70]}

df_raw = pd.DataFrame (raw_data, columns = ['first_name', 'last_name',\
                                           'age', 'preTestScore',\
                                            'postTestScore'])
```

## <font color = grey><b>3. Save DataFrame to CSV file</b></font>

<font color = magenta>
    <b>df.to_csv</b> (csv_filename)
</font>


```python
df_raw.to_csv ("raw.csv")
```

## <font color = grey><b>4. DataFrame Statistics</b></font>

### Information

<font color = magenta><b>df.info ()</b></font>


```python
df_raw.info ()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 5 columns):
     #   Column         Non-Null Count  Dtype 
    ---  ------         --------------  ----- 
     0   first_name     5 non-null      object
     1   last_name      5 non-null      object
     2   age            5 non-null      int64 
     3   preTestScore   5 non-null      object
     4   postTestScore  5 non-null      object
    dtypes: int64(1), object(4)
    memory usage: 328.0+ bytes
    

### Information that indicates how many nulls in each column

E.g. 'Age' value is present only for 714 out of 891 rows.
177 rows do not have value. This count can be achieved by using the function <font color = magenta><b>df.isnull().sum()</b></font> as shown further down.


```python
df_titanic.info ()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    

### Column Names

NOTE: <font color = magenta>columns is an <b>attribute</b></font> that returns an index containing the column names. This attribute can also have a **name**.


```python
df_titanic.columns
```




    Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
          dtype='object')



### Shape

<font color = magenta><b>df.shape</b></font>


```python
df_raw.shape
```




    (5, 5)



### Display first few rows

<font color = magenta><b>df.head (n)</b></font>


```python
df_raw.head (2)
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
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>preTestScore</th>
      <th>postTestScore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jason</td>
      <td>Miller</td>
      <td>42</td>
      <td>4</td>
      <td>25,000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Molly</td>
      <td>Jacobson</td>
      <td>52</td>
      <td>24</td>
      <td>94,000</td>
    </tr>
  </tbody>
</table>
</div>



### Find 'unique' values of a column

<font color = magenta><b>df[col].unique ()</b></font>


```python
df_titanic ['Embarked'].unique ()
```




    array(['S', 'C', 'Q', nan], dtype=object)



### Subtotals, grouped by values of a column

<font color = magenta><b>value_counts ()</b></font> returns a **Series** with <font color = magenta><b>group by values as index values</b></font>.


```python
df_titanic ['Survived'].value_counts ()
```




    0    549
    1    342
    Name: Survived, dtype: int64



NOTE: 'Survived' is used as the field, whose values will be grouped. Hence, the **index is based on the unique values in 'Survived'**.


```python
print(df_titanic ['Survived'].value_counts ().index)
```

    Int64Index([0, 1], dtype='int64')
    

### Extract rows based on a filter

filter = <font color = magenta><b>df [col_filter] == value</b></font> <br>
df <font color = magenta><b>[filter]</b></font>


```python
df_titanic [df_titanic ['Survived'] == 1].head (2)
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



### Subtotals on extracted rows, grouped by values of a column

filter = df [col_filter] == value<br>
df [filter]<font color = magenta><b>[col_groupby].value_counts ()</b></font>


```python
print ('Survived:')
print (df_titanic [df_titanic \
                  ['Survived'] == 1]['Sex'].value_counts ())
print ('\nDidn''t Survive:')
print (df_titanic [df_titanic \
                  ['Survived'] == 0]['Sex'].value_counts ())
```

    Survived:
    female    233
    male      109
    Name: Sex, dtype: int64
    
    Didnt Survive:
    male      468
    female     81
    Name: Sex, dtype: int64
    

**value_counts ()** returns a **Series** with **group by values as index values**.

NOTE: 'Survived' is used as a filter, and 'Sex' is used as the field, whose values will be grouped. Hence, the **index is based on the unique values in 'Sex'**.


```python
type(df_titanic [df_titanic \
                  ['Survived'] == 0]['Sex'].value_counts ())
```




    pandas.core.series.Series



Series has an **index**


```python
df_titanic [df_titanic \
                  ['Survived'] == 0]['Sex'].value_counts ().index
```




    Index(['male', 'female'], dtype='object')



### Null count subtotals

visualizing null values inside the dataframe.<br>
df.<font color = magenta><b>isnull ()</b></font>


```python
df_titanic.isnull ()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>887</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>888</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>889</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>890</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 12 columns</p>
</div>



summary of null values
df.isnull ()<font color = magenta><b>.sum ()</b></font>


```python
df_titanic.isnull ().sum ()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64



## <font color = grey><b>5. Create DataFrame (revisited)</b></font>

### Data creation

#### numpy for NaN
<font color = magenta><b>np.NaN</b></font>


```python
import numpy as np
```

#### Create some lists


```python
author = ['Strang', 'Blitzstein', 'Witten', 'Bishop', 'Bengio', 'Sutton']
title = ['Introduction to Linear Algebra', 'Introduction to Probability', 'ML Beginner', 'ML Advanced', \
         'Deep Learning', 'Reinforcement Learning - An Introduction']
edition = [1, np.NaN, 2, np.NaN, 1, 2]
cost = [10, 20, 15, 40, 30, 25]
topic = ['Maths', 'Maths', 'Machine Learning', 'Machine Learning', \
         'Machine Learning', 'Machine Learning']
sub_topic = ['LA', 'Prob', 'ML', 'ML', 'DL', 'RL']
```

#### Create some Series
<font color = magenta><b>pd.Series (list)</b></font>


```python
s_author = pd.Series (author)
s_title = pd.Series (title)
s_edition = pd.Series (edition)
s_cost = pd.Series (cost)
```

Series has an index. This series has a **range index** <br>
<font color = magenta><b>Series.index</b></font>


```python
s_author.index
```




    RangeIndex(start=0, stop=6, step=1)



### DataFrame from multiple lists

#### Incorrect way


```python
pd.DataFrame ([author, title])
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Strang</td>
      <td>Blitzstein</td>
      <td>Witten</td>
      <td>Bishop</td>
      <td>Bengio</td>
      <td>Sutton</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Introduction to Linear Algebra</td>
      <td>Introduction to Probability</td>
      <td>ML Beginner</td>
      <td>ML Advanced</td>
      <td>Deep Learning</td>
      <td>Reinforcement Learning - An Introduction</td>
    </tr>
  </tbody>
</table>
</div>



The reason authors and titles appear on different rows is that a **list is treated as a column**, and we provided a **list of lists**. So, the outer list of 2 elements (author_list and title_list) created the 2 rows, and, the inner list of authors and titles expanded column-wise.

list is treated as a column


```python
pd.DataFrame (['a', 'b'])
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>



#### Correct way

Zip and create a **list of tuples**. The elements of the **list** form the rows, and the elements of the **enclosing tuple** form the columns. <br>
pd.DataFrame (<font color = magenta><b>list (zip (lst_1, lst_2)) </b></font>, columns = , index = )


```python
pd.DataFrame (list (zip (author, title)), columns = ['Author', 'Title'])
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
      <th>Author</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Witten</td>
      <td>ML Beginner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
    </tr>
  </tbody>
</table>
</div>



Zip and create a **tuple of tuples**. The elements of the **outer tuple** form the rows, and the elements of the **enclosing tuple** form the columns.


```python
pd.DataFrame (tuple (zip (author, title)), columns = ['Author', 'Title'])
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
      <th>Author</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Witten</td>
      <td>ML Beginner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
    </tr>
  </tbody>
</table>
</div>



### DataFrame from a dictionary, using 'lists' as values of keys


```python
frame = {'Author': author, 'Title': title}
df_temp = pd.DataFrame (frame)
df_temp.head ()
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
      <th>Author</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Witten</td>
      <td>ML Beginner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
    </tr>
  </tbody>
</table>
</div>



### DataFrame from a dictionary, using 'Series' as values of keys


```python
frame = {'Author': s_author, 'Title': s_title}
df_books = pd.DataFrame (frame)
df_books.head ()
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
      <th>Author</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Witten</td>
      <td>ML Beginner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
    </tr>
  </tbody>
</table>
</div>



#### Add a 'Series' externally

NOTE:
- A series can be added externally only if the <font color = magenta><b>index</b></font> of the series <font color = magenta><b>matches</b></font> that of the data frame


```python
df_books ['Edition'] = s_edition
df_books.head ()
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
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## <font color = grey><b>6. Index</b></font>

### Provide 'index'

#### Create a custom index


```python
idx = ['a', 'b', 'c', 'd', 'e', 'f']
```

#### Pass nameless indices as a parameter

NOTE:
- '<font color = magenta><b>index</b></font>' is the **row index**
- '<font color = magenta><b>columns</b></font>' is the **column index**


```python
df = pd.DataFrame (tuple (zip (author, title)), \
              columns = ['Author', 'Title'], \
             index = idx)
df
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
      <th>Author</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
    </tr>
    <tr>
      <th>b</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
    </tr>
    <tr>
      <th>c</th>
      <td>Witten</td>
      <td>ML Beginner</td>
    </tr>
    <tr>
      <th>d</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
    </tr>
    <tr>
      <th>e</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
    </tr>
    <tr>
      <th>f</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
    </tr>
  </tbody>
</table>
</div>



#### Provide names for indices

use <font color = magenta><b>rename_axis ()</b></font> to rename index names

A **single parameter implies row index**. <br>
rename_axis (<font color = magenta><b>row_index_name</b></font>)


```python
df2 = df.rename_axis ('Sr. No.')
print ('Row Index Name: ', df2.index.name)
print ('Column Index Name: ', df2.columns.name)
df2
```

    Row Index Name:  Sr. No.
    Column Index Name:  None
    




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
      <th>Author</th>
      <th>Title</th>
    </tr>
    <tr>
      <th>Sr. No.</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
    </tr>
    <tr>
      <th>b</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
    </tr>
    <tr>
      <th>c</th>
      <td>Witten</td>
      <td>ML Beginner</td>
    </tr>
    <tr>
      <th>d</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
    </tr>
    <tr>
      <th>e</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
    </tr>
    <tr>
      <th>f</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
    </tr>
  </tbody>
</table>
</div>



Use **axis** to indicate row (= 0) or column (= 1) axis <br>
rename_axis (<font color = magenta><b>col_axis_name, axis = 1</b></font>)


```python
df2 = df.rename_axis ('Attributes->', axis = 1)
print ('Row Index Name: ', df2.index.name)
print ('Column Index Name: ', df2.columns.name)
df2
```

    Row Index Name:  None
    Column Index Name:  Attributes->
    




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
      <th>Attributes-&gt;</th>
      <th>Author</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
    </tr>
    <tr>
      <th>b</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
    </tr>
    <tr>
      <th>c</th>
      <td>Witten</td>
      <td>ML Beginner</td>
    </tr>
    <tr>
      <th>d</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
    </tr>
    <tr>
      <th>e</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
    </tr>
    <tr>
      <th>f</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
    </tr>
  </tbody>
</table>
</div>



'axis' values <font color = magenta><b>0 and 1</b></font> also carry names "rows" and "columns" respectively.


```python
df2 = df.rename_axis ('Attributes->', axis = "columns")
print ('Row Index Name: ', df2.index.name)
print ('Column Index Name: ', df2.columns.name)
df2
```

    Row Index Name:  None
    Column Index Name:  Attributes->
    




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
      <th>Attributes-&gt;</th>
      <th>Author</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
    </tr>
    <tr>
      <th>b</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
    </tr>
    <tr>
      <th>c</th>
      <td>Witten</td>
      <td>ML Beginner</td>
    </tr>
    <tr>
      <th>d</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
    </tr>
    <tr>
      <th>e</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
    </tr>
    <tr>
      <th>f</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
    </tr>
  </tbody>
</table>
</div>



Rename both row and column axes denoted by **'index' for row index**, and **'columns' for column attribute that returns an index**. <br>
rename_axis (<font color = magenta><b>index</b></font> = row_axis_name, <font color = magenta><b>columns</b></font> = col_axis_name)


```python
df2 = df.rename_axis (index = 'Sr. No.', columns = 'Attributes->')
print ('Row Index Name: ', df2.index.name)
print ('Column Index Name: ', df2.columns.name)
df2
```

    Row Index Name:  Sr. No.
    Column Index Name:  Attributes->
    




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
      <th>Attributes-&gt;</th>
      <th>Author</th>
      <th>Title</th>
    </tr>
    <tr>
      <th>Sr. No.</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
    </tr>
    <tr>
      <th>b</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
    </tr>
    <tr>
      <th>c</th>
      <td>Witten</td>
      <td>ML Beginner</td>
    </tr>
    <tr>
      <th>d</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
    </tr>
    <tr>
      <th>e</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
    </tr>
    <tr>
      <th>f</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
    </tr>
  </tbody>
</table>
</div>



### Hierarchical Indices

#### Create the multi-index

<font color = magenta><b>pd.MultiIndex</b></font>.from_product ()
- list of list of indices
- list of <font color = magenta><b>names</b></font> of indices

NOTE: The list of indices are of different size: There are 3 topics, and 2 sub-topics. So, we have to use <font color = magenta><b>from_product ()</b></font> to perform the cross product of the two indices. This creates a **hierarchy**. If we just wanted two row indices without them being hierarchical, the indices list should be of the same size, and we need to use <font color = magenta><b>from_arrays ()</b></font> instead.


```python
i_topic = ['Mathematics', 'Machine Learning', 'DL/RL']
i_subtopic = [1, 2]
i_names = ['Topic', 'Sub-Topic']

mux = pd.MultiIndex.from_product ([i_topic, i_subtopic], names = i_names)
mux
```




    MultiIndex([(     'Mathematics', 1),
                (     'Mathematics', 2),
                ('Machine Learning', 1),
                ('Machine Learning', 2),
                (           'DL/RL', 1),
                (           'DL/RL', 2)],
               names=['Topic', 'Sub-Topic'])



#### Create the dataframe with multi-index <br>
pd.DataFrame (list, columns = , index = <font color = magenta><b>pd.MultiIndex ().from_#</b></font>)


```python
pd.DataFrame (list (zip (author, title)), \
             columns = ['Author', 'Title'], \
             index = mux)
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Mathematics</th>
      <th>1</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Machine Learning</th>
      <th>1</th>
      <td>Witten</td>
      <td>ML Beginner</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">DL/RL</th>
      <th>1</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
    </tr>
  </tbody>
</table>
</div>



#### Create a multi-index using <font color = magenta><b>from_arrays ()</b></font>


```python
mux = pd.MultiIndex.from_arrays ([topic, sub_topic], names = i_names)
mux
```




    MultiIndex([(           'Maths',   'LA'),
                (           'Maths', 'Prob'),
                ('Machine Learning',   'ML'),
                ('Machine Learning',   'ML'),
                ('Machine Learning',   'DL'),
                ('Machine Learning',   'RL')],
               names=['Topic', 'Sub-Topic'])



#### Create a data frame with multi-index (created using from_arrays ())


```python
pd.DataFrame (list (zip (author, title)), \
             columns = ['Author', 'Title'], \
             index = mux)
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Maths</th>
      <th>LA</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
    </tr>
    <tr>
      <th>Prob</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Machine Learning</th>
      <th>ML</th>
      <td>Witten</td>
      <td>ML Beginner</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
    </tr>
    <tr>
      <th>DL</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
    </tr>
    <tr>
      <th>RL</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
    </tr>
  </tbody>
</table>
</div>



### Navigating using indices

#### Create a dataframe with multi-index


```python
mux = pd.MultiIndex.from_arrays ([topic, sub_topic], names = i_names)
df = pd.DataFrame (list (zip (author, title)), \
             columns = ['Author', 'Title'], \
             index = mux)
df
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Maths</th>
      <th>LA</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
    </tr>
    <tr>
      <th>Prob</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Machine Learning</th>
      <th>ML</th>
      <td>Witten</td>
      <td>ML Beginner</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
    </tr>
    <tr>
      <th>DL</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
    </tr>
    <tr>
      <th>RL</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
    </tr>
  </tbody>
</table>
</div>



#### Filter based on index values

df.<font color = magenta><b>loc [index_val]</b></font> <br>
df.loc [<font color = magenta><b>(index1_val, index2_val)</b></font>]

NOTE: A **data frame is returned**.

df.<font color = magenta><b>sort_index ()</b></font>
- to prevent "PerformanceWarning: **indexing past lexsort depth** may impact performance."


```python
#To prevent 
#"PerformanceWarning: indexing past lexsort depth may impact performance."
df.sort_index (inplace = True)
print ('Books in Mathematics:')
print (df.loc ['Maths'], '\n\n')
print ('Books in Machine Learning:')
print (df.loc ['Machine Learning'], '\n\n')
print ('Books in Deep Learning:')
print (df.loc [('Machine Learning', 'DL')])
```

    Books in Mathematics:
                   Author                           Title
    Sub-Topic                                            
    LA             Strang  Introduction to Linear Algebra
    Prob       Blitzstein     Introduction to Probability 
    
    
    Books in Machine Learning:
               Author                                     Title
    Sub-Topic                                                  
    DL         Bengio                             Deep Learning
    ML         Witten                               ML Beginner
    ML         Bishop                               ML Advanced
    RL         Sutton  Reinforcement Learning - An Introduction 
    
    
    Books in Deep Learning:
                                Author          Title
    Topic            Sub-Topic                       
    Machine Learning DL         Bengio  Deep Learning
    

### Set existing columns as indexes

#### Create a dataframe WITHOUT  passing any index

Instead, **pass index as regular columns** - topic, and sub_topic


```python
#mux = pd.MultiIndex.from_arrays ([topic, sub_topic], names = i_names)
df = pd.DataFrame (list (zip (author, title, topic, sub_topic)), \
             columns = ['Author', 'Title', 'Topic', 'Sub-Topic'], \
             #index = mux
                  )
df
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
      <th>Author</th>
      <th>Title</th>
      <th>Topic</th>
      <th>Sub-Topic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
      <td>Maths</td>
      <td>LA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
      <td>Maths</td>
      <td>Prob</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>Machine Learning</td>
      <td>ML</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
      <td>Machine Learning</td>
      <td>ML</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
      <td>Machine Learning</td>
      <td>DL</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>Machine Learning</td>
      <td>RL</td>
    </tr>
  </tbody>
</table>
</div>



#### Create index from columns

df.<font color = magenta><b>set_index ([cols], inplace = True)</b></font>


```python
df.set_index (['Topic', 'Sub-Topic'], inplace = True)
df
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Maths</th>
      <th>LA</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
    </tr>
    <tr>
      <th>Prob</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Machine Learning</th>
      <th>ML</th>
      <td>Witten</td>
      <td>ML Beginner</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
    </tr>
    <tr>
      <th>DL</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
    </tr>
    <tr>
      <th>RL</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
    </tr>
  </tbody>
</table>
</div>



### Instantiate an index and set it

df.set_index (<font color = magenta><b>pd.RangeIndex (start, stop, step)</b></font>, inplace = True


```python
df.set_index (\
              pd.RangeIndex (start = 5, stop = 11, step = 1),\
              inplace = True)
df
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
      <th>Author</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Witten</td>
      <td>ML Beginner</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
    </tr>
  </tbody>
</table>
</div>



### df.groupby (col)

#### 1. DataFrameGroupBy

df.groupby (<font color = magenta><b>col</b></font>) <br>
df.groupby (<font color = magenta><b>[cols]</b></font>)
 - returns an instance of **DataFrameGroupBy**


```python
df = df_titanic
df.groupby (['Survived', 'Sex'])
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x000001CAEA053A30>



#### 2. Summary Statistics

Perform a **summary statistic** on DataFrameGroupBy
- mean () - for numerical columns
- count () - for all columns
- <font color = magenta><b>size ()</b></font> - for numerical columns

NOTE:
- stats are obviously not performed on groupby columns, since those column values are used to form groups
- **count** returns **individual column count of rows having some value**
> count of columns like 'Age' and 'Cabin' indicate missing values
- **sum** returns **count of rows**. some column having missing value does not impact, unless all columns have missing values.


```python
print ('Information of the data frame:')
print (df.info (), '\n\n')
print ('Mean Statistics:')
print (df.groupby (['Survived', 'Sex']).mean (), '\n\n')
print ('Count Statistics:')
print (df.groupby (['Survived', 'Sex']).count (), '\n\n')
print ('Size Statistics:')
print (df.groupby (['Survived', 'Sex']).size ())
```

    Information of the data frame:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    None 
    
    
    Mean Statistics:
                     PassengerId    Pclass        Age     SibSp     Parch  \
    Survived Sex                                                            
    0        female   434.851852  2.851852  25.046875  1.209877  1.037037   
             male     449.121795  2.476496  31.618056  0.440171  0.207265   
    1        female   429.699571  1.918455  28.847716  0.515021  0.515021   
             male     475.724771  2.018349  27.276022  0.385321  0.357798   
    
                          Fare  
    Survived Sex                
    0        female  23.024385  
             male    21.960993  
    1        female  51.938573  
             male    40.821484   
    
    
    Count Statistics:
                     PassengerId  Pclass  Name  Age  SibSp  Parch  Ticket  Fare  \
    Survived Sex                                                                  
    0        female           81      81    81   64     81     81      81    81   
             male            468     468   468  360    468    468     468   468   
    1        female          233     233   233  197    233    233     233   233   
             male            109     109   109   93    109    109     109   109   
    
                     Cabin  Embarked  
    Survived Sex                      
    0        female      6        81  
             male       62       468  
    1        female     91       231  
             male       45       109   
    
    
    Size Statistics:
    Survived  Sex   
    0         female     81
              male      468
    1         female    233
              male      109
    dtype: int64
    

#### > Series (with indexes) returned by groupby.summary_stat

NOTE:
- df.groupby (col).size () <font color = magenta>returns a <b>series</b></font>, with an **index**
- df.groupby (**[cols]**).size () returns a series with a <font color = magenta><b>multi-index</b></font>

#### >> groupby (**col**)

**series**


```python
df.groupby ('Survived').size ()
```




    Survived
    0    549
    1    342
    dtype: int64



type = <font color = magenta><b>Series</b></font>


```python
type(df.groupby ('Survived').size ())
```




    pandas.core.series.Series



<font color = magenta><b>index</b></font> of the series


```python
df.groupby ('Survived').size ().index
```




    Int64Index([0, 1], dtype='int64', name='Survived')



#### >> groupby (**[cols]**)

NOTE:
- didn't survive 549 = 81 + 468
- survived       342 = 233 + 109

**series**


```python
df.groupby (['Survived', 'Sex']).size ()
```




    Survived  Sex   
    0         female     81
              male      468
    1         female    233
              male      109
    dtype: int64



type = **series**


```python
type (df.groupby (['Survived', 'Sex']).size ())
```




    pandas.core.series.Series



<font color = magenta><b>multi-index</b></font> of the series


```python
df.groupby (['Survived', 'Sex']).size ().index
```




    MultiIndex([(0, 'female'),
                (0,   'male'),
                (1, 'female'),
                (1,   'male')],
               names=['Survived', 'Sex'])



#### 3. Unstack the multi-index series into a data frame

You may wish to **remove** one of the features **from the hierarchical index** and <font color = magenta><b>form different columns with respect to that feature</b></font>. You can do so using the <font color = magenta><b>unstack</b></font> method.

NOTE:
unstack () requires the feature to have <font color = magenta><b>unique values</b></font>.
This is because the values of the feature form column names.
An index created using <font color = magenta><b>groupby () ensures</b></font> that the index has <font color = magenta><b>unique values</b></font>.


```python
df.groupby (['Survived', 'Sex']).size ().unstack ()
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
      <th>Sex</th>
      <th>female</th>
      <th>male</th>
    </tr>
    <tr>
      <th>Survived</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>81</td>
      <td>468</td>
    </tr>
    <tr>
      <th>1</th>
      <td>233</td>
      <td>109</td>
    </tr>
  </tbody>
</table>
</div>



NOTE:
- didn't survive 549 = 81 + 468
- survived       342 = 233 + 109

The **index column to unstack**, is passed as parameter, can be given by name, or by position, 0 is the default.


```python
df.groupby (['Survived', 'Sex']).size ().unstack ('Sex')
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
      <th>Sex</th>
      <th>female</th>
      <th>male</th>
    </tr>
    <tr>
      <th>Survived</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>81</td>
      <td>468</td>
    </tr>
    <tr>
      <th>1</th>
      <td>233</td>
      <td>109</td>
    </tr>
  </tbody>
</table>
</div>



by default, the <font color = magenta><b>leaf of the hierarchical (multi) index is unstacked</b></font>, leaving the rest of the index tree as the index of the returned dataframe


```python
df.groupby (['Survived', 'Sex']).size ().unstack ().index
```




    Int64Index([0, 1], dtype='int64', name='Survived')



the unstacked leaf of the index forms the columns of the returned dataframe


```python
df.groupby (['Survived', 'Sex']).size ().unstack ().columns
```




    Index(['female', 'male'], dtype='object', name='Sex')



**mention the index column to unstack**

NOTE
- this time we have <font color = magenta><b>unstacked 'Survived' instead of 'Sex'</b></font>


```python
df.groupby (['Survived', 'Sex']).size ().unstack ('Survived')
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
      <th>Survived</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>81</td>
      <td>233</td>
    </tr>
    <tr>
      <th>male</th>
      <td>468</td>
      <td>109</td>
    </tr>
  </tbody>
</table>
</div>



## <font color = grey><b>7. Filter</b></font>

### Data creation


```python
frame = {'Author': s_author, 'Title': s_title, 'Edition': s_edition}
df_books = pd.DataFrame (frame)
df_books.head ()
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
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



### An example 'filter'


```python
df_books ['Edition'] == 1
```




    0     True
    1    False
    2    False
    3    False
    4     True
    5    False
    Name: Edition, dtype: bool



#### <font color = magenta><b>Chaining</b></font> format


```python
df_books.Edition.eq (2)
```




    0    False
    1    False
    2     True
    3    False
    4    False
    5     True
    Name: Edition, dtype: bool



### Apply filter to a dataframe

#### Filter rows whose column value equals a specific value


```python
filter = df_books ['Edition'] == 2
df_books [filter]
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
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



##### Filter rows whose column value does NOT equal a specific value - <font color = magenta><b>!=</b></font>


```python
filter = df_books.Edition != 2
df_books [filter]
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
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Filter rows whose column value is not NaN - df.col.<font color = magenta><b>notnull ()</b></font>


```python
filter = df_books.Edition.notnull ()
df_books [filter]
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
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Filter rows with column values among a set of values

df.colname.<font color = magenta><b>isin</b></font>


```python
authors_maths = ['Apostol', 'Strang', 'Blitzstein']
filter = df_books.Author.isin (authors_maths)
df_books [filter]
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
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



##### Filter rows with column values NOT among a set of values

<font color = magenta><b>~df</b></font>.colname.isin


```python
filter = ~df_books.Author.isin (authors_maths)
df_books [filter]
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
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Filter rows based on <font color = magenta><b>multiple conditions</b></font>

condition1 <font color = magenta><b>&</b></font> condition2


```python
filter = ~df_books.Author.isin (authors_maths)\
        & df_books.Edition.notnull ()
df_books [filter]
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
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Filter rows based on a <font color = magenta><b>series</b></font>


```python
s_title_learning = df_books ['Title'\
                            ].apply (lambda x: x.endswith ('Learning'))
df_books [s_title_learning]
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
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## <font color = grey><b>8. Selection</b></font>

### using <font color = magenta><b>.loc</b></font>
- selection using <font color = magenta><b>label-based location</b></font>
- selection using <font color = magenta><b>filter</b></font>

df.loc [(<font color = magenta><b>index1_label, ...</b></font>), [selected_cols]]

df.loc [<font color = magenta><b>[index_label1, index_label2, ...]</b></font>, [selected_cols]]

df.loc [<font color = magenta><b>filter</b></font>, [selected_cols]]

NOTE: For the first parameter of .loc (), <font color = magenta><b>we pass index values</b></font>. These values are the **labels of the rows**. Hence, .loc is label-based.

#### data creation - add index


```python
frame = {'Author': s_author, 'Title': s_title, 'Edition': s_edition}
df_books = pd.DataFrame (frame)
mux = pd.MultiIndex.from_arrays ([topic, sub_topic],\
                                 names = ['Topic', 'Sub-Topic'])
df_books = df_books.set_index (mux)
#to prevent
#"PerformanceWarning: indexing past lexsort depth may impact performance."
df_books.sort_index (inplace = True)
df_books
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Machine Learning</th>
      <th>DL</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>RL</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Maths</th>
      <th>LA</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Prob</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### df.loc [<font color = magenta><b>index</b></font>] - returns rows belonging to the index label


```python
df_books.loc ['Machine Learning']
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
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
    <tr>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DL</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>RL</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



#### df.loc [<font color = magenta><b>(index1, index2)</b></font>] - returns rows belonging to the <font color = magenta><b>hierarchical index labels</b></font>


```python
df_books.loc [('Machine Learning', 'ML')]
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Machine Learning</th>
      <th>ML</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



df.loc [<font color = magenta><b>[idx1_labl1, idx1_lab2]</b></font>] - returns rows belonging to the multiple labels of the <font color = magenta><b>index root</b></font>


```python
df_books.loc [['Machine Learning', 'Maths']]
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Machine Learning</th>
      <th>DL</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>RL</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Maths</th>
      <th>LA</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Prob</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



df.loc [([idx1_lab1, idx1_lab2]<font color = magenta><b>, [idx2_lab1]</b></font>), [selected_cols]] - returns rows belonging to the <font color = magenta><b>cross-product labels</b></font> of the nodes of the hierarchical index


```python
df_books.loc [(['Machine Learning', 'Maths'],\
               ['LA', 'DL', 'RL']), ['Title']]
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
      <th></th>
      <th>Title</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Machine Learning</th>
      <th>DL</th>
      <td>Deep Learning</td>
    </tr>
    <tr>
      <th>RL</th>
      <td>Reinforcement Learning - An Introduction</td>
    </tr>
    <tr>
      <th>Maths</th>
      <th>LA</th>
      <td>Introduction to Linear Algebra</td>
    </tr>
  </tbody>
</table>
</div>



#### df.loc [(index1, index2)<font color = magenta><b>, col_name</b></font>] - returns rows belonging to the label, and their <font color = magenta><b>single column</b></font>

returns a <font color = magenta><b>series</b></font>


```python
df_books.loc [('Machine Learning', 'ML'), 'Title']
```




    Topic             Sub-Topic
    Machine Learning  ML           ML Beginner
                      ML           ML Advanced
    Name: Title, dtype: object



#### df.loc [(index1, index2), <font color = magenta><b>[</b></font>col_name<font color = magenta><b>]</b></font> - returns rows belonging to the label, and their <font color = magenta><b>one or more columns</b></font>

returns a <font color = magenta><b>data frame</b></font>


```python
df_books.loc [('Machine Learning', 'ML'), ['Title']]
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
      <th></th>
      <th>Title</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Machine Learning</th>
      <th>ML</th>
      <td>ML Beginner</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>ML Advanced</td>
    </tr>
  </tbody>
</table>
</div>



#### df.loc [(index1, index2), <font color = magenta><b>[cols]</b></font>] - returns rows belonging to the label, and their <font color = magenta><b>one or more columns</b></font>


```python
df_books.loc [('Machine Learning', 'ML'), ['Title', 'Edition']]
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
      <th></th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Machine Learning</th>
      <th>ML</th>
      <td>ML Beginner</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>ML Advanced</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### df.loc [<font color = magenta><b>filter</b></font>] - returns rows <font color = magenta><b>satisfying the filter</b></font>


```python
filter = df_books ['Author'] == 'Sutton'
df_books.loc [filter]
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Machine Learning</th>
      <th>RL</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



NOTE:
- <font color = magenta><b>df [filter]</b></font> and <font color = magenta><b>df.loc [filter]</b></font> both return the <font color = red><b>same result</b></font>.
- But, **df.loc** provides <font color = magenta><b>selection of columns</b></font>


```python
df_books [filter]
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Machine Learning</th>
      <th>RL</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_books.loc [filter, ['Edition', 'Author']]
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
      <th></th>
      <th>Edition</th>
      <th>Author</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Machine Learning</th>
      <th>RL</th>
      <td>2.0</td>
      <td>Sutton</td>
    </tr>
  </tbody>
</table>
</div>



#### df.loc [<font color = magenta><b>series</b></font>] - return rows based on a <font color = magenta><b>series</b></font>

Create a series by applying a <font color = magenta><b>lambda on a column</b></font>


```python
s_title_learning = df_books ['Title'\
                            ].apply (lambda x: x.endswith ('Learning'))
df_books.loc [s_title_learning, ['Edition', 'Title']]
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
      <th></th>
      <th>Edition</th>
      <th>Title</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Machine Learning</th>
      <th>DL</th>
      <td>1.0</td>
      <td>Deep Learning</td>
    </tr>
  </tbody>
</table>
</div>



### using <font color = magenta><b>.iloc</b></font> - integer-location-based location

df.iloc [row_range, col_range]

NOTE: For both the parameters of .iloc (), <font color = magenta><b>we pass location indices</b></font>. Hence, .iloc is integer-location-based.

NOTE:
- a **single** row or column is returned as a <font color = magenta><b>series</b></font>
> have returned a dataframe by <font color = magenta>enclosing the index inside a <b>[]</b></font>
- **multiple** rows or columns are returned as a <font color = magenta><b>dataframe</b></font>

#### df.iloc [<font color = magenta><b>int_location</b></font>] returns a row as <font color = magenta><b>series</b></font>


```python
df_books.iloc [0]
```




    Author            Bengio
    Title      Deep Learning
    Edition                1
    Name: (Machine Learning, DL), dtype: object



#### df.iloc [<font color = magenta><b>[int_location]</b></font>] returns a row as <font color = magenta><b>data frame</b></font>


```python
df_books.iloc [[0]]
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Machine Learning</th>
      <th>DL</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### df.iloc [<font color = magenta><b>row_range</b></font>] - returns multiple rows from the range


```python
df_books.iloc [2:5]
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Machine Learning</th>
      <th>ML</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>RL</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Maths</th>
      <th>LA</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### df.iloc [row_range<font color = magenta><b>, col_range</b></font>] - returns multiple rows and columns from the range


```python
df_books.iloc [2:5, 1:3]
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
      <th></th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Machine Learning</th>
      <th>ML</th>
      <td>ML Advanced</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>RL</th>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Maths</th>
      <th>LA</th>
      <td>Introduction to Linear Algebra</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### df.iloc [row_range<font color = magenta><b>, [-1]</b></font>] - returns multiple rows from the range, and the <font color = magenta><b>last column</b></font>


```python
df_books.iloc [2:5, [-1]]
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
      <th></th>
      <th>Edition</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Machine Learning</th>
      <th>ML</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>RL</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Maths</th>
      <th>LA</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### get the <font color = magenta><b>first row</b></font>


```python
df_books.iloc [[0]]
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Machine Learning</th>
      <th>DL</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### get the <font color = magenta><b>last row</b></font>


```python
df_books.iloc [[-1]]
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Maths</th>
      <th>Prob</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## <font color = grey><b>9. Update</b></font>

### Data Creation


```python
frame = {'Author': s_author, 'Title': s_title,\
         'Edition': s_edition}
df_books = pd.DataFrame (frame)
mux = pd.MultiIndex.from_arrays ([topic, sub_topic],\
                                 names = ['Topic', 'Sub-Topic'])
df_books = df_books.set_index (mux)
#to prevent
#"PerformanceWarning: indexing past lexsort depth may impact performance."
df_books.sort_index (inplace = True)
df_books
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Machine Learning</th>
      <th>DL</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>RL</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Maths</th>
      <th>LA</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Prob</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### Add a 'cost column

NOTE:
- A series can be added externally only if the <font color = magenta><b>index</b></font> of the series <font color = magenta><b>matches</b></font> that of the data frame
- This can be achieved by pd.Series (lst, <font color = magenta><b>index = </b></font>)


```python
cost_indexed = [30, np.NaN, 40, 25, np.NaN, 20]
df_books ['Cost'] = pd.Series (cost_indexed, index = df_books.index)
df_books
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
      <th>Cost</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Machine Learning</th>
      <th>DL</th>
      <td>Bengio</td>
      <td>Deep Learning</td>
      <td>1.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
      <td>NaN</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>RL</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Maths</th>
      <th>LA</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Prob</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
      <td>NaN</td>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
</div>



### Using <font color = magenta><b>.loc</b></font>


```python
df_books.loc [('Machine Learning', ['DL']), ['Title']]\
= 'Deep Reinforcement Learning'
df_books
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
      <th>Cost</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Machine Learning</th>
      <th>DL</th>
      <td>Bengio</td>
      <td>Deep Reinforcement Learning</td>
      <td>1.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
      <td>NaN</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>RL</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Maths</th>
      <th>LA</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Prob</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
      <td>NaN</td>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
</div>



### <font color = magenta><b>fill missing values</b></font>

#### fill with a <font color = magenta><b>default</b></font> value

Note:
- df[col].<font color = magenta><b>fillna ()</b></font> updates a column (not inplace) and returns it as a <font color = magenta><b>series</b></font>. Pandas Philosophy: single row/column returned as series.
- The returned series is assigned to a dataframe column in the LHS


```python
df_books ['Edition'] = df_books ['Edition'].fillna ('0.0')
df_books
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
      <th>Cost</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Machine Learning</th>
      <th>DL</th>
      <td>Bengio</td>
      <td>Deep Reinforcement Learning</td>
      <td>1</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
      <td>0.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>RL</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Maths</th>
      <th>LA</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Prob</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
      <td>0.0</td>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
</div>



#### fill with a <font color = magenta><b>stats</b></font> value

> <font color = magenta><b>groupby based on index</b></font> - <font color = magenta><b>level =</b></font> [l1, l2, ..]


```python
df_books.groupby (level = [0])['Cost'].sum ()
```




    Topic
    Machine Learning    95.0
    Maths               20.0
    Name: Cost, dtype: float64



> <font color = magenta><b>median value</b></font> of the group


```python
df_books.groupby (level = [0])['Cost'].transform ('median')
```




    Topic             Sub-Topic
    Machine Learning  DL           30.0
                      ML           30.0
                      ML           30.0
                      RL           30.0
    Maths             LA           20.0
                      Prob         20.0
    Name: Cost, dtype: float64



> missing values filled with median value of the group
NOTE:
- values updated for '<font color = magenta><b>ML Beginner</b></font>' and '<font color = magenta><b>Introduction to Linear Algebra</b></font>'


```python
#series of median cost grouped by index level 0 ('Topic')
s_cost_median_by_group = df_books.groupby (level = [0])\
['Cost'].transform ('median')
#update the empty cells of 'Cost'
df_books ['Cost'].fillna (s_cost_median_by_group, inplace = True)
df_books
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
      <th>Cost</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Machine Learning</th>
      <th>DL</th>
      <td>Bengio</td>
      <td>Deep Reinforcement Learning</td>
      <td>1</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>2</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
      <td>0.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>RL</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Maths</th>
      <th>LA</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
      <td>1</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>Prob</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
      <td>0.0</td>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
</div>



### Using <font color = magenta><b>columns</b></font>


```python
df_books ['Description'] = df_books ['Title'] + ' - by - '\
+ df_books ['Author']
df_books
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
      <th>Cost</th>
      <th>Description</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Machine Learning</th>
      <th>DL</th>
      <td>Bengio</td>
      <td>Deep Reinforcement Learning</td>
      <td>1</td>
      <td>30.0</td>
      <td>Deep Reinforcement Learning - by - Bengio</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>2</td>
      <td>30.0</td>
      <td>ML Beginner - by - Witten</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>ML Advanced - by - Bishop</td>
    </tr>
    <tr>
      <th>RL</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2</td>
      <td>25.0</td>
      <td>Reinforcement Learning - An Introduction - by ...</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Maths</th>
      <th>LA</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
      <td>1</td>
      <td>20.0</td>
      <td>Introduction to Linear Algebra - by - Strang</td>
    </tr>
    <tr>
      <th>Prob</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>Introduction to Probability - by - Blitzstein</td>
    </tr>
  </tbody>
</table>
</div>



### <font color = magenta><b>binning</b></font> using <font color = magenta><b>pd.cut ()</b></font>

binning - **numeric to categorical**


```python
cost_boundaries = [0, 11, 21, 31, 41, np.inf]
cost_range = ['Free', 'Cheap', 'Affordable', \
              'Costly', 'Exorbitant']
df_books ['Cost Range'] = pd.cut \
(df_books ['Cost'], cost_boundaries, labels = cost_range)
df_books
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
      <th></th>
      <th>Author</th>
      <th>Title</th>
      <th>Edition</th>
      <th>Cost</th>
      <th>Description</th>
      <th>Cost Range</th>
    </tr>
    <tr>
      <th>Topic</th>
      <th>Sub-Topic</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Machine Learning</th>
      <th>DL</th>
      <td>Bengio</td>
      <td>Deep Reinforcement Learning</td>
      <td>1</td>
      <td>30.0</td>
      <td>Deep Reinforcement Learning - by - Bengio</td>
      <td>Affordable</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Witten</td>
      <td>ML Beginner</td>
      <td>2</td>
      <td>30.0</td>
      <td>ML Beginner - by - Witten</td>
      <td>Affordable</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>Bishop</td>
      <td>ML Advanced</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>ML Advanced - by - Bishop</td>
      <td>Costly</td>
    </tr>
    <tr>
      <th>RL</th>
      <td>Sutton</td>
      <td>Reinforcement Learning - An Introduction</td>
      <td>2</td>
      <td>25.0</td>
      <td>Reinforcement Learning - An Introduction - by ...</td>
      <td>Affordable</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Maths</th>
      <th>LA</th>
      <td>Strang</td>
      <td>Introduction to Linear Algebra</td>
      <td>1</td>
      <td>20.0</td>
      <td>Introduction to Linear Algebra - by - Strang</td>
      <td>Cheap</td>
    </tr>
    <tr>
      <th>Prob</th>
      <td>Blitzstein</td>
      <td>Introduction to Probability</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>Introduction to Probability - by - Blitzstein</td>
      <td>Cheap</td>
    </tr>
  </tbody>
</table>
</div>


