---
layout: post
title: "NumPy Playground"
date: 2020-12-02 22:32:59 +0530
categories:
---

NumPy or <font color = magenta><b>Numeric Python</b></font> is a package for computation on **homogenous** n-dimensional arrays.

Uses:
- perform **operations on all the elements** of two list **directly**.

# A. <font color = orange><b>Imports</b></font>


```python
#array handling
import numpy as np

#random sampling from distributions
from numpy.random import randn, normal, standard_normal

#plotting
from matplotlib import pyplot as plt
import seaborn as sns
```

# B. <font color = orange><b>Preliminaries</b></font>

## 1. row vector, column vector, and matrix

All are of type numpy.<font color = magenta><b>ndarray</b></font>


```python
row_v = np.array ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # row vector (1-D)
# column vector (2-D)
col_v = np.array ([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
mat_2d = np.array ([[1, 'a'], [2, 'b'], [3, 'c']]) # 2-D matrix
mat_3d = np.array ([[[1, 'a'], [2, 'b'], [3, 'c']], \
                   [[4, 'd'], [5, 'e'], [6, 'f']], \
                   [[7, 'g'], [8, 'h'], [9, 'i']]]) # 3-D matrix

print ('row_v shape:', row_v.shape, '\n', row_v, '\n',\
       'element (1) of 1D array:', row_v[1], '\n')
print ('col_v shape:', col_v.shape, '\n', col_v, '\n',\
       'element (2,0) of 2D array:', col_v[2][0], '\n')
print ('mat_2d shape:', mat_2d.shape, '\n', mat_2d, '\n',\
       'element (2,1) of 2D array:', mat_2d[2][1], '\n')
print ('mat_3d shape:', mat_3d.shape, '\n', mat_3d, '\n',\
       'element (2,1,0) of 3D array:', mat_3d[2][1][0],\
       'element (2,2,1) of 3D array:', mat_3d[2][2][1],'\n')
```

    row_v shape: (10,) 
     [0 1 2 3 4 5 6 7 8 9] 
     element (1) of 1D array: 1 
    
    col_v shape: (10, 1) 
     [[0]
     [1]
     [2]
     [3]
     [4]
     [5]
     [6]
     [7]
     [8]
     [9]] 
     element (2,0) of 2D array: 2 
    
    mat_2d shape: (3, 2) 
     [['1' 'a']
     ['2' 'b']
     ['3' 'c']] 
     element (2,1) of 2D array: c 
    
    mat_3d shape: (3, 3, 2) 
     [[['1' 'a']
      ['2' 'b']
      ['3' 'c']]
    
     [['4' 'd']
      ['5' 'e']
      ['6' 'f']]
    
     [['7' 'g']
      ['8' 'h']
      ['9' 'i']]] 
     element (2,1,0) of 3D array: 8 element (2,2,1) of 3D array: i 
    
    

## 2. np.<font color = magenta><b>zeros</b></font>, np.<font color = magenta><b>ones</b></font>,  and np.<font color = magenta><b>full</b></font>

### <font color = magenta><b>numerical</b></font> arrays

np.**zeros** (**shape**, **dtype** = int/float) <br>
np.**ones** (**shape**, **dtype** = int/float)


```python
print (type (np.zeros ((1, 2), dtype = int))) #type
print ('integer 1D ndarray:\n',\
       np.zeros ((3), dtype = int), '\n') #integer 1D ndarray
print ('integer 2D ndarray:\n',\
       np.ones ((1, 2), dtype = int), '\n') #integer 2D ndarray
print ('float 2D ndarray:\n',\
       np.zeros ((1, 2), dtype = float), '\n') #float 2D ndarray
print ('float 3D ndarray:\n',\
       np.ones ((2, 3, 2), dtype = float), '\n') #float 3D ndarray
```

    <class 'numpy.ndarray'>
    integer 1D ndarray:
     [0 0 0] 
    
    integer 2D ndarray:
     [[1 1]] 
    
    float 2D ndarray:
     [[0. 0.]] 
    
    float 3D ndarray:
     [[[1. 1.]
      [1. 1.]
      [1. 1.]]
    
     [[1. 1.]
      [1. 1.]
      [1. 1.]]] 
    
    

### <font color = magenta><b>boolean</b></font> arrays

np.<font color = red><b>zeros</b></font> (shape, dtype = <font color = red><b>bool</b></font>)


```python
print ('boolean 2D ndarray:\n',\
       np.ones ((1, 2), dtype = bool), '\n')
print ('boolean 3D ndarray:\n',\
       np.ones ((2, 3, 2), dtype = bool), '\n')
```

    boolean 2D ndarray:
     [[ True  True]] 
    
    boolean 3D ndarray:
     [[[ True  True]
      [ True  True]
      [ True  True]]
    
     [[ True  True]
      [ True  True]
      [ True  True]]] 
    
    

### <font color = magenta><b>any-type</b></font> arrays

np.<font color = magenta><b>full</b></font> (shape, **value**)

> **Deduce type** from value

2D **boolean** array


```python
np.full ((2,3), False)
```




    array([[False, False, False],
           [False, False, False]])



2D **integer** array


```python
np.full ((2,3), 7)
```




    array([[7, 7, 7],
           [7, 7, 7]])



## 3. np.<font color = magenta><b>arange</b></font>, and np.<font color = magenta><b>linspace</b></font>

The essential difference between NumPy linspace and NumPy arange is that linspace enables you to <font color = magenta><b>control the precise end value</b></font>, whereas arange gives you <font color = magenta><b>more direct control over the increments between values</b></font> in the sequence.

np.<font color = magenta><b>arange</b></font> (**start** = , **stop** = , **step** = )

- only '**stop**' is **mandatory**


```python
print ('arange (10): stop at 10', np.arange (10))
print ('arange (-1, 10, 2): start at -1,\n \
stop at 10, step size = 2:', np.arange (-1, 10, 2))
```

    arange (10): stop at 10 [0 1 2 3 4 5 6 7 8 9]
    arange (-1, 10, 2): start at -1,
     stop at 10, step size = 2: [-1  1  3  5  7  9]
    

np.<font color = magenta><b>linspace</b></font> (**start** = , **stop** = , **num** = )

- creates sequences of **evenly spaced** values within a defined interval
- num **includes the endpoints**


```python
np.linspace (0, 100, 5)
```




    array([  0.,  25.,  50.,  75., 100.])



## 4. <font color = magenta><b>Structured arrays</b></font> and <font color = magenta><b>Field Access</b></font>

Structured arrays are ndarrays whose datatype is a <font color = magenta><b>composition of simpler datatypes</b></font> organized as a sequence of named fields.

If the ndarray object is a structured array the fields of the array can be accessed by <font color = magenta><b>indexing the array with strings, dictionary-like</b></font>.

Returns a new <font color = magenta><b>view</b></font> to the array


```python
x = np.array ([('Bishop', 1, 44.99), ('Bengio', 2, 39.99),\
               ('Sutton', 2, 24.99)],\
              dtype = [('Author', 'U10'), ('Edition', 'i4'),\
                       ('Price', 'f4')])
print ('x:\n', x)
print ('\nx.shape:\n', x.shape)
print ('\nx [2]:\n', x [2])
print ('\nx ["Author"]:\n', x ['Author'])
print ('\nx ["Author"].shape: same as x.shape:\n', x ['Author'].shape)
print ('\nx ["Price"] = 19.99')
x ['Price'] = 19.99
print ('\nx ["Price"]:\n', x ['Price'])
```

    x:
     [('Bishop', 1, 44.99) ('Bengio', 2, 39.99) ('Sutton', 2, 24.99)]
    
    x.shape:
     (3,)
    
    x [2]:
     ('Sutton', 2, 24.99)
    
    x ["Author"]:
     ['Bishop' 'Bengio' 'Sutton']
    
    x ["Author"].shape: same as x.shape:
     (3,)
    
    x ["Price"] = 19.99
    
    x ["Price"]:
     [19.99 19.99 19.99]
    

**Structured datatypes** are designed to be able to mimic ‘structs’ in the C language, and share a similar memory layout. They are <font color = blue><b>meant for interfacing with C code</b></font> and for low-level manipulation of structured buffers, for example for interpreting binary blobs. For these purposes they support specialized features such as subarrays, nested datatypes, and unions, and allow control over the memory layout of the structure.

Users looking to manipulate tabular data, such as stored in **csv files**, may find other pydata projects more suitable, such as xarray, <font color = magenta><b>pandas</b></font>, or DataArray. These provide a high-level interface for tabular data analysis and are better optimized for that use. For instance, the C-struct-like memory layout of structured arrays in numpy can lead to poor cache behavior in comparison.


```python
x = np.zeros((2,2), dtype=[('a', np.int32), ('b', np.float64, (3,3))])
print ('x:\n', x)
print ('\nx.shape:\n', x.shape)
print ('Indexing x["field-name"] returns a new view to the array,\
which is of the same shape as x (except when the field is a sub-array) ')
print ('\nx ["a"]:\n', x ['a'])
print ('\nx ["a"].shape: same as x.shape:\n', x ['a'].shape)
print ('\nx ["b"]:\n', x ['b'])
print ('\nx ["b"].shape: NOT same as x.shape:\n', x ['b'].shape)
```

    x:
     [[(0, [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
      (0, [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])]
     [(0, [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
      (0, [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])]]
    
    x.shape:
     (2, 2)
    Indexing x["field-name"] returns a new view to the array,which is of the same shape as x (except when the field is a sub-array) 
    
    x ["a"]:
     [[0 0]
     [0 0]]
    
    x ["a"].shape: same as x.shape:
     (2, 2)
    
    x ["b"]:
     [[[[0. 0. 0.]
       [0. 0. 0.]
       [0. 0. 0.]]
    
      [[0. 0. 0.]
       [0. 0. 0.]
       [0. 0. 0.]]]
    
    
     [[[0. 0. 0.]
       [0. 0. 0.]
       [0. 0. 0.]]
    
      [[0. 0. 0.]
       [0. 0. 0.]
       [0. 0. 0.]]]]
    
    x ["b"].shape: NOT same as x.shape:
     (2, 2, 3, 3)
    

# C. <font color = orange><b>Array Properties</b></font>

### shape

> ndarray.**shape**


```python
print ('Shape of 2D matrix: ', mat_2d.shape)
print ('Shape of 3D matrix: ', mat_3d.shape)
```

    Shape of 2D matrix:  (3, 2)
    Shape of 3D matrix:  (3, 3, 2)
    

### size

> ndarray.**size**


```python
print ('Size of 2D matrix: ', mat_2d.size)
print ('Size of 3D matrix: ', mat_3d.size) 
```

### number of dimensions

> ndarray.ndim


```python
print ('Dimensions of 2D matrix: ', mat_2d.ndim)
print ('Dimensions of 3D matrix: ', mat_3d.ndim) 
```

- <font color = red><b>TODO TODO TODO</b></font>: ndarray.flags, dtype, itemsize, strides

# D. <font color = orange><b>Array Broadcasting</b></font>

Operations between **differently sized arrays** is called <font color = magenta><b>broadcasting</b></font>

Operations between **same sized arrays** is called <font color = magenta><b>vectorization</b></font>

## 1. Introduction

NumPy provides a mechanism for performing mathematical <font color = magenta><b>operations</b></font> on arrays of <font color = magenta><b>unequal shapes</b></font>.

#### an example

> an example of a **(3, 4) * (4, )** multiplication


```python
x_2d = np.array ([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]])
y_1d = np.array ([1, 2, 3, 4])
print ('x.shape:', x_2d.shape)
print ('y.shape:', y_1d.shape)
print ('(x * y).shape:\n', (x_2d * y_1d).shape)
print ('x * y:\n', x_2d * y_1d)
```

#### Rules of Broadcasting

To determine if two arrays are broadcast-compatible, align the entries of their shapes such that their trailing dimensions are aligned, and then check that each pair of aligned dimensions satisfy either of the following conditions:

- the <font color = magenta><b>aligned dimensions</b></font> have the <font color = magenta><b>same size</b></font>
- one of the dimensions has a <font color = magenta><b>size of 1</b></font>

The two arrays are broadcast-compatible if **either of these conditions** are satisfied for each pair of aligned dimensions.

Broadcasting is not reserved for operations between 1-D and 2-D arrays, and furthermore **both arrays in an operation may undergo broadcasting**. That being said, not all pairs of arrays are broadcast-compatible.

#### Intuition

- perform broadcasting of an array to a higher dimension <br>
- use this intuition of broadcasting to understand how it happens when performing mathemarical operations on multiple arrays <br>
- np.<font color = magenta><b>broadcast_to</b></font> (df, dim_tuple)


```python
y = np.array([[ 0],
                [ 1],
            [-1]])
print (y.shape)
np.broadcast_to (y, (3, 3, 2))
```

#### an example

> an example of a **(3, 1, 2) * (3, 1)** multiplication <br>
> (3, 1, 2) <br>
> ....(3, 1) <br>
> --------- <br>
> (3, 3, 2) <br>
> the second rule of broadcasting (one of the dimensions has a size of 1) is applicable


```python
x = np.array([[[0, 1]],
            [[2, 3]],
            [[4, 5]]])
print ('x.shape:', x.shape)
print ('y.shape:', y.shape)
print ('(x * y).shape:', (x * y).shape)
print ('x * y:\n', x * y)
```

## 2. Inserting Size-1 Dimensions into An Array

To tailor the shape <font color = magenta><b>for broadcasting</b></font>

### using <font color = magenta><b>reshape</b></font>


```python
print ('row_v:', row_v)
print ('row_v.shape:', row_v.shape)
print ('reshape.shape:', (row_v.reshape (1, row_v.shape[0], 1, 1)).shape)
print ('reshaped:\n', row_v.reshape (1, row_v.shape[0], 1, 1))
```

> <font color = magenta><b>TODO TODO TODO</b></font> np.ravel - numpy.ravel( M[ : , 0] ) -- converts shape from (R, 1) to (R,)

### using np.<font color = magenta><b>newaxis</b></font>

> ndarray [np.newaxis<font color = magenta><b>, :, </b></font>np.newaxis, np.newaxis]


```python
print ('row_v:', row_v)
print ('row_v.shape:', row_v.shape)
print ('newaxis.shape:', row_v [np.newaxis, :,\
                                np.newaxis, np.newaxis].shape)
print ('reshaped:\n', row_v [np.newaxis, :,\
                             np.newaxis, np.newaxis])
```

## 3. Using size-1 dimension to make arrays broadcasting compatible


```python
x_1d = np.array([1, 2, 3])
print ('x_1d.shape:', x_1d.shape)
print ('x_1d:\n', x_1d)
x_2d = x_1d.reshape (x_1d.shape[0], 1)
print ('\nx_2d.shape:', x_2d.shape)
print ('x_2d:\n', x_2d)
y_1d = np.array ([4, 5, 6, 7])
print ('\ny_1d.shape:', y_1d.shape)
print ('y_1d:\n', y_1d)

#ValueError: operands could not be broadcast together 
# with shapes (3,) (4,)
#print (x_1d * y_1d)

print ('\n(x_2d * y_1d).shape:', (x_2d * y_1d).shape)
print ('x_2d * y_1d:\n', x_2d * y_1d)
print ('\ny_1d * x_2d:\n', y_1d * x_2d)
```

> An interesting application **Pairwise Distances** can be found at https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/Broadcasting.html#Size-1-Axes-&-The-newaxis-Object

# E. <font color = orange><b>Indexing</b></font>

Note:
- The best way to think about **NumPy arrays** is that they consist of **two parts**, a <font color = magenta><b>data buffer</b></font> which is just a block of raw elements, and a <font color = magenta><b>view</b></font> which describes how to interpret the data buffer.
- https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r

## 1. Selection Object

ndarray [<font color = magenta><b>selection_object</b></font>]
selection_object can be
- <font color = magenta><b>integers</b></font>
- a <font color = magenta><b>selection</b></font> tuple
- a <font color = magenta><b>slice</b></font> object
- a <font color = magenta><b>Ellipsis</b></font> object
- a <font color = magenta><b>numpy.newaxis</b></font> object
- a <font color = magenta><b>non-tuple sequence</b></font> object

## 2. Kinds of indexing

There are three kinds of indexing available:
- <font color = magenta><b>basic slicing</b></font>
    - integers
    - slice object
- <font color = magenta><b>advanced indexing</b></font>
- <font color = magenta><b>field access</b></font> <br>

Which one occurs <font color = magenta><b>depends on the selection object</b></font>.

Note:
- the rules of **basic indexing** specifically call for a **tuple of indices**. Supplying a **list of indices** <font color = magenta><b>triggers advanced indexing</b></font> rather than basic indexing!


```python
print ('2D Array:\n', mat_2d)
print ('Using Basic Indexing [(1, -1)]:\n',\
       mat_2d [(1, -1)])
print ('Using Advanced Indexing [[1, -1]]:\n',\
       mat_2d [[1, -1]])
print ('Using Advanced Indexing [[1, -1], [-1]]:\n',\
       mat_2d [[1, -1], [-1]])
```

### 2.1 basic slicing

Note:
- the rules of **basic indexing** specifically call for a **tuple of indices**. Supplying a **list of indices** <font color = magenta><b>triggers advanced indexing</b></font> rather than basic indexing!

All arrays generated by basic slicing are always <font color = magenta><b>views</b></font> of the original array.

Note: <br>
NumPy slicing creates a view **instead of a copy** as in the case of builtin Python sequences such as string, tuple and list. Care must be taken when extracting a small portion from a large array which becomes useless after the extraction, because the small portion extracted contains a reference to the large original array whose memory will not be released until all arrays derived from it are **garbage-collected**. In such cases an explicit **copy() is recommended**.

#### 2.1.1. selection object is <font color = magenta><b>integers</b></font> or <font color = magenta><b>tuple</b></font>

Note:
- In Python, x<font color = magenta><b>[(</b></font>exp1, exp2, ..., expN<font color = magenta><b>)]</b></font> is equivalent to x<font color = magenta><b>[</b></font>exp1, exp2, ..., expN<font color = magenta><b>]</b></font>; the latter is just <font color = magenta><b>syntactic sugar</b></font> for the former.
- (exp1, exp2, ..., expN) is a <font color = magenta><b>selection tuple</b></font>.


```python
print (mat_2d [(2, 1)])
print (mat_2d [2, 1]) #syntactic sugar
print (mat_2d [2][1]) #is this too syntactic sugar? TODO
print (mat_3d [(0,1,1)])
```

#### 2.1.2. selection object is a <font color = magenta><b>slice object</b></font>

> ndarray [<font color = magenta><b>slice object</b></font>] <br>
> - ndarray [**start** : **stop** : **step**] <br>
> - 'start : stop : step' is a **slice object** <br>
> - <font color = magenta><b>class slice</b></font> (start, stop[, step])


```python
print ('row vector:', row_v)
start = 2
stop = 8
step = 2
print ('Elements of row vector starting at ',\
       start, 'with step size of ', step,\
       '\n ending at', stop, ': ',\
       row_v [start : stop : step])
```

> <font color = magenta><b>slicing syntax forms a slice object</b></font> behind the scenes


```python
print (row_v [start : stop : step])
print (row_v [slice (start, stop, step)])
```

> <font color = magenta><b>negative</b></font> start / stop / step

Note:
- Negative 'start' and 'stop' are interpreted as **n + start** and **n + stop** where n is the number of elements in the corresponding dimension.
- Negative 'step' makes **stepping go towards smaller indices**.


```python
print (row_v)
print ('neg_start pos_stop',  row_v [-5 : 8])
print ('neg_start neg_stop',  row_v [-5 : -2])
print ('neg_start neg_stop neg_step',  row_v [-2 : -5: -2])
```

> <font color = magenta><b>default</b></font> indices

Note:
- Assume n is the number of elements in the dimension being sliced. Then, if i is not given it defaults to 0 for k > 0 and n - 1 for k < 0 . If j is not given it defaults to n for k > 0 and <font color = red><b>-n-1</b></font> for k < 0 . If k is not given it defaults to 1.
- <font color = magenta><b>::</b></font> is the same as <font color = magenta><b>:</b></font> and means select all indices along this axis.


```python
# if i is not given it defaults to 0 for k > 0
print ('if i is not given it defaults to 0 for k > 0:',\
       row_v [:8:1])
# if i is not given it defaults to n - 1 for k < 0
print ('if i is not given it defaults to n - 1 for k < 0:',\
       row_v [:8:-1])
# If j is not given it defaults to n for k > 0
print ('If j is not given it defaults to n for k > 0:',\
       row_v [0::1])
# If j is not given it defaults to -n-1?? (or -1?) for k < 0
print ('If j is not given it defaults to -n-1?? (or -1?) for k < 0:',\
       row_v [5::-1])
# If k is not given it defaults to 1
print ('If k is not given it defaults to 1:',\
       row_v[2:7])
print ('ndarray [::]:', row_v [::])
print ('ndarray [:]:', row_v [::])
```

#### 2.1.3. selection object is a <font color = magenta><b>tuple of integers and slice objects</b></font>


```python
print ('mat_2d:\n', mat_2d)
print ('mat_2d.shape:\n', mat_2d.shape)
print ('mat_2d [0:3:2, 1]:', mat_2d [0:3:2, 1])
```

> An integer, i, returns the same values as i:i+1 except the <font color = magenta><b>dimensionality</b></font> of the returned object is <font color = magenta><b>reduced by 1</b></font>.


```python
print ('row_v:', row_v)
print ('row_v.shape:', row_v.shape)
print ('row_v [2]:', row_v [2])
print ('row_v [2].shape:', row_v [2].shape)
print ('row_v [2:3]:', row_v [2:3])
print ('row_v [2:3].shape:', row_v [2:3].shape)
```

#### 2.1.4. selection object is a <font color = magenta><b>tuple of Ellipsis and slice objects</b></font>


```python
print ('mat_2d:\n', mat_2d)
print ('mat_2d.shape:\n', mat_2d.shape)
print ('all rows, col_0: mat_2d [..., 0:1:1]:\n', mat_2d [..., 0:1:1])
print ('shape:', mat_2d [..., 0:1:1].shape)
```

#### 2.1.5. selection object is a <font color = magenta><b>tuple of Ellipsis and integers</b></font>

Ellipsis **expands to** the number of : objects needed for the **selection tuple to index all dimensions**. In most cases, this means that <font color = magenta><b>length of the expanded selection tuple is x.ndim</b></font>. There may only be a single ellipsis present.


```python
print ('mat_2d:\n', mat_2d)
print ('mat_2d.shape:\n', mat_2d.shape)
print ('all rows, col_1: mat_2d [..., 1]:\n', mat_2d [..., 1])
print ('all rows, col_1: mat_2d [:, 1]:\n', mat_2d [:, 1])
print ('shape:', mat_2d [..., 1].shape)
print ('Note that the last dimension unfolded.')
print ('Here the shape (3,) means the array is indexed \
by a single index which runs from 0 to 2')

print ('\n\nmat_3d:\n', mat_3d)
print ('mat_3d.shape:\n', mat_3d.shape)
print ('all rows of rows, col_1: mat_3d [..., 1]:\n', mat_3d [..., 1])
print ('all rows of rows, col_1: mat_3d [:, :, 1]:\n', mat_3d [:, :, 1])
print ('shape:', mat_3d [..., 1].shape)
print ('Note that the last dimension unfolded.')

print ('\n\nmat_3d:\n', mat_3d)
print ('mat_3d.shape:\n', mat_3d.shape)
print ('all rows, col_0 of dim2, col_1 of dim3: mat_3d [..., 0, 1]:\n',\
       mat_3d [..., 0, 1])
print ('all rows, col_0 of dim2, col_1 of dim3: mat_3d [:, 0, 1]:\n',\
       mat_3d [:, 0, 1])
print ('shape:', mat_3d [..., 0, 1].shape)
print ('Note that both the second-last and last dimensions unfolded.')
```

#### 2.1.6. selection object is the <font color = magenta><b>newaxis object</b></font>

np.<font color = magenta><b>newaxis</b></font>

Each newaxis object in the selection tuple serves to <font color = magenta><b>expand the dimensions</b></font> of the resulting selection by one unit-length dimension. The added dimension is the position of the newaxis object in the selection tuple.


```python
print ('mat_2d:\n', mat_2d)
print ('mat_2d.shape:\n', mat_2d.shape)
print ('\nmat_2d [1]:\n', mat_2d [1],\
       'shape:', mat_2d [1].shape)
print ('Expanded dimension: mat_2d [np.newaxis, 1]:\n',\
       mat_2d [np.newaxis, 1],\
       'shape:', mat_2d [np.newaxis, 1].shape)

print ('\nmat_2d [:, 1]:\n', mat_2d [:, 1],\
      'shape:', mat_2d [:, 1].shape)
print ('Expanded dimension: mat_2d [:, 1, np.newaxis]:\n',\
       mat_2d [:, 1, np.newaxis],\
      'shape:', mat_2d [:, 1, np.newaxis].shape)
```

np.newaxis can be used to **add a dimension to an extracted row or column**, <font color = magenta><b>to avoid calling ndarray.reshape()</b></font>, when making matrices size compatible for various operations.
-  https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r


```python
print ('mat_2d:\n', mat_2d)
print ('mat_2d.shape:\n', mat_2d.shape)

print ('\nfetch a row: mat_2d [1]:\n', mat_2d [1])
print ('Note the trailing "," : mat_2d [1].shape:', mat_2d [1].shape)
print ('fetch a row: mat_2d [np.newaxis, 1]:\n', mat_2d [np.newaxis, 1],\
       'shape:', mat_2d [np.newaxis, 1].shape)

print ('\nfetch a column: mat_2d [:, 1]:\n', mat_2d [:, 1])
print ('Note the trailing "," : mat_2d [:, 1].shape:',\
       mat_2d [:, 1].shape)
print ('fetch a column: mat_2d [:, 1, np.newaxis]:\n',\
       mat_2d [:, 1, np.newaxis],\
      'shape:', mat_2d [:, 1, np.newaxis].shape)

```

#### 2.1.7 Basic slicing <font color = magenta><b>extends</b></font> Python’s basic concept of slicing to <font color = magenta><b>N dimensions</b></font>.


```python
print ('2D array:\n', mat_2d)
print ('first two elements of col_1:\n', mat_2d [0:2:1, 1:2:1])
# 0:2:1 - selects rows [0,1]
# 1:2:1 - selects cols [1]
print ('all two elements of col_1:\n', mat_2d [:, 1:2:1])
```

### 2.2 advanced indexing

#### 2.2.1 introduction

Advanced indexing is triggered when the selection object is:
- a non-tuple **sequence** object
- an **ndarray** (of data type integer or bool)
- **tuple** with at least one
    - sequence object, or
    - ndarray (of data type integer or bool).

Advanced indexing always returns a <font color = magenta><b>copy of the data</b></font> (contrast with basic slicing that returns a view)

Note:
- the rules of **basic indexing** specifically call for a **tuple of indices**. Supplying a **list of indices** <font color = magenta><b>triggers advanced indexing</b></font> rather than basic indexing!


```python
print ('2D Array:\n', mat_2d)
print ('Using Basic Indexing [(1, -1)]:\n',\
       mat_2d [(1, -1)])
print ('Using Advanced Indexing [[1, -1]]:\n',\
       mat_2d [[1, -1]])
print ('Using Advanced Indexing [[1, -1], [-1]]:\n',\
       mat_2d [[1, -1], [-1]])
```

The definition of advanced indexing means that **x[(1,2,3),]** is fundamentally different than **x[(1,2,3)]**. The latter is equivalent to x[1,2,3] which will trigger basic selection while the former will trigger advanced indexing. It is a <font color = magenta><b>sequence</b></font>.


```python
x = np.array ([[0, 1, 2], [3, 4, 5]])
print ('x:', x)
print ('\nTuple as a selection object: x[(0, 1)]:\n', x[(0, 1)])
print ('\nSequence as a selection object: x[(0, 1), ]:\n', x[(0, 1), ])
print ('\nSequence as a selection object: x[(0, 1), (2), ]:\n', x[(0, 1), (2), ])
```

#### 2.2.2 integer array indexing

**example**


```python
print ('mat_3d:\n', mat_3d)
print ('\nmat_3d [[0, 1], [1], [1]]:\n', mat_3d [[0, 1], [1], [1]])
```

**example** - get corner elements


```python
x = np.array([[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8],
            [ 9, 10, 11]])
print (x, '\n')
rows = [[0, 0], [3, 3]]
cols = [[0, 2], [0, 2]]
print (x [rows, cols])
```

**example** - get corner elements - using <font color = magenta><b>broadcasting</b></font> - using np.<font color = magenta><b>ix_</b></font>


```python
rows = np.array ([0, 3])
cols = np.array ([0, 2])
print ('rows:', rows)
print ('cols', cols)
print ('\nindex for broadcasting: rows [:, np.newaxis]:\n',\
       rows [:, np.newaxis])
print ('rows [:, np.newaxis].shape: ', rows [:, np.newaxis].shape)
print ('cols.shape:', cols.shape)
print ('broadcasting index will be (2, 2)')
print ('\nx [rows [:, np.newaxis], cols]:\n',\
       x [rows [:, np.newaxis], cols])
print ('\nx [np.ix_ (rows, cols)]:\n', x [np.ix_ (rows, cols)])
```

#### 2.2.3 boolean array indexing

#### ndarray.<font color = magenta><b>nonzero</b></font> ()


```python
x = np.array ([1, 2, 0, 7, 4, 2, 0, 6, 8])
print ('x:\n', x)
print ('x.nonzero () returns a tuple:', x.nonzero ())
print ('get non-zero elements:', x[x.nonzero ()])
```

np.<font color = magenta><b>isnan</b></font> (ndarray)


```python
x = np.array([[1., 2.], [np.nan, 3.], [np.nan, np.nan]])
print ('x:\n', x)
print ('\nnp.isnan (x):\n', np.isnan (x))
#bool_array = np.isnan (x)
#print ('bool_array:\n', bool_array)
#idx = bool_array.nonzero ()
#print ('idx:\n', idx)
print ('\nx [~np.isnan (x)]:\n', x [~np.isnan (x)])
```

#### ndarray [<font color = magenta><b>condition</b></font>]


```python
x = np.array ([1, 2, 0, 7, 4, 2, 0, 6, 8])
print ('x:\n', x)
print ('\nx [x < 7]:\n', x [x < 7])
```

In general if an index includes a Boolean array, the result will be **identical to inserting sel_obj.nonzero()** into the same position and using the integer array indexing mechanism

x[ind_1, boolean_array, ind_2] is equivalent to x[(ind_1,) + **boolean_array.nonzero()** + (ind_2,)].


```python
x = np.array ([1, 2, 0, 7, 4, 2, 0, 6, 8])
print ('x:\n', x)
print ('\nx < 5:\n', x < 5)
print ('\nx < 5:\n', (x < 5).nonzero ())
print ('\nx [(x < 5).nonzero ()]:\n', x [(x < 5).nonzero ()])
```

select all **rows** which **sum up to less or equal** two


```python
x = np.array([[0, 1], [1, 1], [2, 2]])
print ('x:\n', x)
print ('\nx.shape:\n', x.shape)
rowsum = x.sum (-1)
print ('\nrowsum: x.sum (-1): \n', rowsum)
print ('\nx [rowsum <= 2, :]:\n', x [rowsum <= 2, :])
```

<font color = red><b>TODO</b></font> if **rowsum** would have **two dimensions** as well - <font color = magenta><b>keepdims</b></font>

x = np.array([[0, 1], [1, 1], [2, 2]])
print ('x:\n', x)
print ('\nx.shape:\n', x.shape)
rowsum = x.sum (-1, keepdims = True)
print ('\nrowsum: x.sum (-1): \n', rowsum)
print ('\nrowsum.shape \n', rowsum.shape)
#print ('\nrowsum.nonzero() \n', rowsum.nonzero ())
print ('\nx [rowsum <= 2]: no need of ", :"\n')
print (x [rowsum <= 2])

**x [condition] += n**

Note:
- x is on LHS, so, the origiinal x gets replaced by the copy that advanced indexing created


```python
x = np.array ([1, -2, 0, 7, -4, 2, 0, -6, 8])
print ('x:\n', x)
print ('\nx [x < 0] += 20')
x [x < 0] += 20
print ('\nx:\n', x)
```

### 2.3 field access

Refer '<font color = magenta><b>Structured arrays and Field Access</b></font>' in the preliminaries section.

## 3. flat iterator indexing

x.flat returns an iterator that will iterate over the entire array. This iterator object can also be indexed using **basic slicing** or **advanced indexing** as long as the **selection object is not a tuple**. This should be clear from the fact that x.flat is a <font color = magenta><b>1-dimensional view</b></font>.

https://www.geeksforgeeks.org/numpy-indexing/


```python
x = np.array([[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8],
            [ 9, 10, 11]])

print ('x:\n', x)
print ('\nx [0:2]:\n', x [0:2])
print ('\nx.flat [0:2]:\n', x.flat [0:2])
```

# F. <font color = orange><b>Vectorized Operations</b></font>

Operations between **differently sized arrays** is called <font color = magenta><b>broadcasting</b></font>

Operations between **same sized arrays** is called <font color = magenta><b>vectorization</b></font>

## a. Introduction

https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/VectorizedOperations.html

### 1. data


```python
x = np.array([[ 0.,  1.,  2.],
            [ 3.,  4.,  5.],
            [ 6.,  7.,  8.]])

y = np.array([[-4. , -3.5, -3. ],
            [-2.5, -2. , -1.5],
            [-1. , -0.5, -0. ]])
```

The examples that follow are based on a **taxonomy** of:
- operations on elements of a single array
- operations on 'corresponding' elements of two arrays
- operations on elements of a single array using the same scalar
- summary operations on elements of a single array

Refer '<font color = blue><b>Taxonomy of Operations</b></font>' further below for another taxonomy.

### 2. example of <font color = magenta><b>operations on entries</b></font> of a single array

> <font color = red><b>unary</b></font> operations? <br>
> Operations on a single array are <font color = magenta><b>not necessarily unary</b></font>
> > example: the **logical (binary) operation** x < 6

Arithmetic operations **with scalars** are as you would expect, **propagating** the value <font color = magenta><b>to each element</b></font>

#### square - this is a <font color = magenta><b>binary</b></font> operation : x ** y


```python
x ** 2
```




    array([[ 0.,  1.,  4.],
           [ 9., 16., 25.],
           [36., 49., 64.]])



#### np.<font color = magenta><b>sqrt</b></font> (ndarray)


```python
np.sqrt (x)
```




    array([[0.        , 1.        , 1.41421356],
           [1.73205081, 2.        , 2.23606798],
           [2.44948974, 2.64575131, 2.82842712]])



operations on a <font color = magenta><b>slice of an array</b></font>


```python
print ('x:\n', x)
print ('\nAdd 0.5 to the second column')
print ('\n0.5 + x [:, 1]:\n', 0.5 + x [:, 1])
```

    x:
     [[0. 1. 2.]
     [3. 4. 5.]
     [6. 7. 8.]]
    
    Add 0.5 to the second column
    
    0.5 + x [:, 1]:
     [1.5 4.5 7.5]
    

<font color = magenta><b>logical operation</b></font> - this is a <font color = magenta><b>binary</b></font> operation : x < y


```python
print ('x < 5:\n', x < 5)
```

    x < 5:
     [[ True  True  True]
     [ True  True False]
     [False False False]]
    

### 3. mathematical operations performed between <font color = magenta><b>two arrays</b></font> are designed to act on the <font color = magenta><b>corresponding pairs of entries</b></font> between the two arrays

> <font color = magenta><b>binary</b></font> operations?

#### <font color = magenta><b>+</b></font> operator


```python
print ('x:\n', x)
print ('\ny:\n', y)
print ('\nx + y:\n', x + y)
```

    x:
     [[0. 1. 2.]
     [3. 4. 5.]
     [6. 7. 8.]]
    
    y:
     [[-4.  -3.5 -3. ]
     [-2.5 -2.  -1.5]
     [-1.  -0.5 -0. ]]
    
    x + y:
     [[-4.  -2.5 -1. ]
     [ 0.5  2.   3.5]
     [ 5.   6.5  8. ]]
    

#### <font color = magenta><b>*</b></font> operator


```python
print ('x:\n', x)
print ('\ny:\n', y)
print ('\nx * y:\n', x * y)
```

    x:
     [[0. 1. 2.]
     [3. 4. 5.]
     [6. 7. 8.]]
    
    y:
     [[-4.  -3.5 -3. ]
     [-2.5 -2.  -1.5]
     [-1.  -0.5 -0. ]]
    
    x * y:
     [[-0.  -3.5 -6. ]
     [-7.5 -8.  -7.5]
     [-6.  -3.5 -0. ]]
    

<font color = red><b>(minus) 0 ?</b></font> in x*y [0][0]

np.<font color = magenta><b>dot</b></font> and np.<font color = magenta><b>multiply</b></font>

This function returns the dot product of two arrays. For **1-D** arrays, it is the <font color = magenta><b>inner product</b></font> of the vectors.


```python
np.dot(np.array([1, -3, 4]), np.array([2, 0, 1]))
```




    6



For **2-D** vectors, it is the **equivalent to matrix multiplication**. np.<font color = magenta><b>matmul</b></font>


```python
print ('x:\n', x)
print ('\ny:\n', y)
print ('\nnp.dot (x, y):\n', np.dot (x, y))
print ('\nnp.matmul (x, y):\n', np.matmul (x, y))
```

    x:
     [[0. 1. 2.]
     [3. 4. 5.]
     [6. 7. 8.]]
    
    y:
     [[-4.  -3.5 -3. ]
     [-2.5 -2.  -1.5]
     [-1.  -0.5 -0. ]]
    
    np.dot (x, y):
     [[ -4.5  -3.   -1.5]
     [-27.  -21.  -15. ]
     [-49.5 -39.  -28.5]]
    
    np.matmul (x, y):
     [[ -4.5  -3.   -1.5]
     [-27.  -21.  -15. ]
     [-49.5 -39.  -28.5]]
    

**Is matmul a vectorized operation?**

np.<font color = magenta><b>multiply</b></font>


```python
print ('x:\n', x)
print ('\ny:\n', y)
print ('\nnp.multiply (x, y):\n', np.multiply (x, y))
```

    x:
     [[0. 1. 2.]
     [3. 4. 5.]
     [6. 7. 8.]]
    
    y:
     [[-4.  -3.5 -3. ]
     [-2.5 -2.  -1.5]
     [-1.  -0.5 -0. ]]
    
    np.multiply (x, y):
     [[-0.  -3.5 -6. ]
     [-7.5 -8.  -7.5]
     [-6.  -3.5 -0. ]]
    

### 4. Operate on <font color = magenta><b>sequences of numbers</b></font>

<font color = magenta><b>Sequential functions</b></font> can act on an array’s entries **as if they form a single sequence**, or act on subsequences of the array’s entries, according to the array’s axes.

#### <font color = magenta><b>sum</b></font>


```python
print ('x:\n', x)
print ('\nsum (x):\n', sum (x))
print ('\nnp.sum (x):\n', np.sum (x))
print ('\nnp.sum (x, axis = 0):\n', np.sum (x, axis = 0))
print ('\nnp.sum (x, axis = 1):\n', np.sum (x, axis = 1))
```

    x:
     [[0. 1. 2.]
     [3. 4. 5.]
     [6. 7. 8.]]
    
    sum (x):
     [ 9. 12. 15.]
    
    np.sum (x):
     36.0
    
    np.sum (x, axis = 0):
     [ 9. 12. 15.]
    
    np.sum (x, axis = 1):
     [ 3. 12. 21.]
    

Why is **python sum** <font color = red><b>summing up columns</b></font>? , and returning ndarray too?

## b. Taxonomy of Operations

### i. <font color = magenta><b>Mathematical</b></font> Operations

#### 1. Unary Functions: <font color = magenta><b>f (x)</b></font>

#### np.<font color = magenta><b>sqrt</b></font> (ndarray)


```python
np.sqrt (x)
```




    array([[0.        , 1.        , 1.41421356],
           [1.73205081, 2.        , 2.23606798],
           [2.44948974, 2.64575131, 2.82842712]])



#### np.<font color = magenta><b>log</b></font> (ndarray) - ln (x)


```python
np.log (x)
```

    /opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: RuntimeWarning: divide by zero encountered in log
      """Entry point for launching an IPython kernel.
    




    array([[      -inf, 0.        , 0.69314718],
           [1.09861229, 1.38629436, 1.60943791],
           [1.79175947, 1.94591015, 2.07944154]])



#### np.<font color = magenta><b>exp</b></font> (ndarray) - e^x


```python
np.exp (x)
```




    array([[1.00000000e+00, 2.71828183e+00, 7.38905610e+00],
           [2.00855369e+01, 5.45981500e+01, 1.48413159e+02],
           [4.03428793e+02, 1.09663316e+03, 2.98095799e+03]])



#### 2. Binary Functions: <font color = magenta><b>f (x, y)</b></font>

There are two cases that we must consider when working with binary functions, in the context of NumPy arrays:

- When **both operands** of the function are **arrays** (of the same shape).
- When one operand of the function is a <font color = magenta><b>scalar</b></font> (i.e. a single number) and the other is an **array**.

#### <font color = magenta><b>+</b></font> operator


```python
print ('x:\n', x)
print ('\ny:\n', y)
print ('\nx + y:\n', x + y)
print ('\nScalar operand: x + 5:\n', x + 5)
```

    x:
     [[0. 1. 2.]
     [3. 4. 5.]
     [6. 7. 8.]]
    
    y:
     [[-4.  -3.5 -3. ]
     [-2.5 -2.  -1.5]
     [-1.  -0.5 -0. ]]
    
    x + y:
     [[-4.  -2.5 -1. ]
     [ 0.5  2.   3.5]
     [ 5.   6.5  8. ]]
    
    Scalar operand: x + 5:
     [[ 5.  6.  7.]
     [ 8.  9. 10.]
     [11. 12. 13.]]
    

#### square - this is a <font color = magenta><b>binary</b></font> operation : x ** y and np.<font color = magenta><b>power</b></font> - one operand is a <font color = magenta><b>scalar</b></font>


```python
print ('x:\n', x)
print ('\nx ** 2:\n', x ** 2)
print ('\nnp.power (x, 3):\n', np.power (x, 3))
```

    x:
     [[0. 1. 2.]
     [3. 4. 5.]
     [6. 7. 8.]]
    
    x ** 2:
     [[ 0.  1.  4.]
     [ 9. 16. 25.]
     [36. 49. 64.]]
    
    np.power (x, 3):
     [[  0.   1.   8.]
     [ 27.  64. 125.]
     [216. 343. 512.]]
    

np.<font color = magenta><b>maximum</b></font>


```python
print ('x:\n', x)
print ('\ny:\n', y)
print ('\nnp.maximum (x, y):\n', np.maximum (x, y))
```

    x:
     [[0. 1. 2.]
     [3. 4. 5.]
     [6. 7. 8.]]
    
    y:
     [[-4.  -3.5 -3. ]
     [-2.5 -2.  -1.5]
     [-1.  -0.5 -0. ]]
    
    np.maximum (x, y):
     [[0. 1. 2.]
     [3. 4. 5.]
     [6. 7. 8.]]
    

#### 3. functions that operate on sequence of numbers: <font color = magenta><b>f ({x_i} i = 0..n-1)</b></font>

#### np.<font color = magenta><b>mean</b></font>
#### np.<font color = magenta><b>median</b></font>
#### np.<font color = magenta><b>var</b></font> - variance
#### np.<font color = magenta><b>std</b></font> - standard deviation
#### np.<font color = magenta><b>max</b></font> - see binary function '<font color = blue><b>np.maximum</b></font>' for array of maximum elements among corresponding elements of the two arrays
#### np.<font color = magenta><b>min</b></font> - minimum element of the array
#### np.<font color = magenta><b>argmax</b></font> - index (sequential) of the maximum element of the array
#### np.<font color = magenta><b>argmin</b></font>
#### np.<font color = magenta><b>sum</b></font>


```python
print ('x:\n', x)
print ('\nnp.mean (x):\n', np.mean (x))
print ('\nnp.median (x):\n', np.median (x))
print ('\nnp.var (x):\n', np.var (x))
print ('\nnp.std (x):\n', np.std (x))
print ('\nnp.max (x):\n', np.max (x))
print ('\nnp.min (x):\n', np.min (x))
print ('\nnp.argmax (x):\n', np.argmax (x))
print ('\nnp.argmin (x):\n', np.argmin (x))
print ('\nnp.sum (x):\n', np.sum (x))
```

    x:
     [[0. 1. 2.]
     [3. 4. 5.]
     [6. 7. 8.]]
    
    np.mean (x):
     4.0
    
    np.median (x):
     4.0
    
    np.var (x):
     6.666666666666667
    
    np.std (x):
     2.581988897471611
    
    np.max (x):
     8.0
    
    np.min (x):
     0.0
    
    np.argmax (x):
     8
    
    np.argmin (x):
     0
    
    np.sum (x):
     36.0
    

##### 3.1 np.func (ndarray, <font color = magenta><b>axis</b></font> = )


```python
print ('x:\n', x)
print ('\nnp.sum (x, axis = 0):\n', np.sum (x, axis = 0))
print ('\nnp.sum (x, axis = 1):\n', np.sum (x, axis = 1))
```

    x:
     [[0. 1. 2.]
     [3. 4. 5.]
     [6. 7. 8.]]
    
    np.sum (x, axis = 0):
     [ 9. 12. 15.]
    
    np.sum (x, axis = 1):
     [ 3. 12. 21.]
    

another **example**


```python
x = np.arange(24).reshape(4,2,3)
print ('x:\n', x)
print ('\nx.shape:\n', x.shape)
print ('\nnp.sum (x, axis = 0):\n', np.sum (x, axis = 0))
print ('\nnp.sum (x, axis = 1):\n', np.sum (x, axis = 1))
print ('\nnp.sum (x, axis = 2):\n', np.sum (x, axis = 2))
print ('\nnp.sum (x, axis = (0, 1)):\n', np.sum (x, axis = (0,1)))
print ('2+5+8+11+14+17+20+23 = :', 2+5+8+11+14+17+20+23)
```

    x:
     [[[ 0  1  2]
      [ 3  4  5]]
    
     [[ 6  7  8]
      [ 9 10 11]]
    
     [[12 13 14]
      [15 16 17]]
    
     [[18 19 20]
      [21 22 23]]]
    
    x.shape:
     (4, 2, 3)
    
    np.sum (x, axis = 0):
     [[36 40 44]
     [48 52 56]]
    
    np.sum (x, axis = 1):
     [[ 3  5  7]
     [15 17 19]
     [27 29 31]
     [39 41 43]]
    
    np.sum (x, axis = 2):
     [[ 3 12]
     [21 30]
     [39 48]
     [57 66]]
    
    np.sum (x, axis = (0, 1)):
     [ 84  92 100]
    2+5+8+11+14+17+20+23 = : 100
    

### ii. <font color = magenta><b>Logical</b></font> Operations

#### Binary operations

**data**


```python
x = np.array([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [12, 13, 14, 15]])
```

**example**


```python
print ('x:\n', x)
print ('\nx.shape:\n', x.shape)
print ('\nx < 6:\n', x < 6)
```

    x:
     [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
    
    x.shape:
     (4, 4)
    
    x < 6:
     [[ True  True  True  True]
     [ True  True False False]
     [False False False False]
     [False False False False]]
    

#### Sequence operations

np.<font color = magenta><b>allclose</b></font>

You should **never rely on two floating point numbers being exactly equal**. Rather, you should require that they are sufficiently “close” in value. In this same vein, you ought not check that the entries of two float-type arrays are precisely equal. Towards this end, the function allclose can be used to verify that all corresponding pairs of entries between two arrays are approximately equal in value


```python
x = np.array([0.1, 0.2, 0.3])
y = np.array([1., 2., 3.]) / 10
print ('x:\n', x)
print ('\ny:\n', y)
print ('\nnp.allclose(x, y):\n', np.allclose(x, y))
```

    x:
     [0.1 0.2 0.3]
    
    y:
     [0.1 0.2 0.3]
    
    np.allclose(x, y):
     True
    

## c. Performance of Vectorized Operations

measure time using <font color = magenta><b>timeit</b></font>


```python
import timeit
element_count = 10000000 #how many elements in the array
thread_invoc_count = 10 #how many times to run the thread

#data
x = np.random.standard_normal (element_count)

#thread callables
def python_sum ():
    return sum (x)
def python_loop_sum ():
    sum = 0
    for i in x:
        sum += i
    return sum
def numpy_sum ():
    return np.sum (x)

#threads
t1 = timeit.timeit (python_sum, number = thread_invoc_count)
t2 = timeit.timeit (python_loop_sum, number = thread_invoc_count)
t3 = timeit.timeit (numpy_sum, number = thread_invoc_count)

#call threads
print ('python_sum thread took', t1, 'seconds.')
print ('python_loop_sum thread took', t2, 'seconds.')
print ('numpy_sum thread took', t3, 'seconds.')
```

    python_sum thread took 21.519882301000052 seconds.
    python_loop_sum thread took 29.515229176000048 seconds.
    numpy_sum thread took 0.06145643499985454 seconds.
    

with:
element_count = 10000000
thread_invoc_count = 10
- **python_sum** thread took <font color = red><b>21.53</b></font> seconds.
- **python_loop_sum** thread took <font color = red><b>29.51</b></font> seconds.
- **numpy_sum** thread took <font color = magenta><b>0.06</b></font> seconds.

# G. <font color = orange><b>Linear Algebra</b></font>

https://becominghuman.ai/an-essential-guide-to-numpy-for-machine-learning-in-python-5615e1758301

**multiple linear regression** - http://www2.lawrence.edu/fast/GREGGJ/Python/numpy/numpyLA.html

np.<font color = magenta><b>linalg</b></font>

**data**


```python
#Create a Matrix
matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
print('\nmatrix:\n', matrix)
```

    
    matrix:
     [[1 2 3]
     [4 5 6]
     [7 8 9]]
    

ndarray<font color = magenta><b>.T</b></font> - **transpose** - np.<font color = magenta><b>transpose</b></font>


```python
print('\nmatrix:\n', matrix)
print('\nmatrix.T:\n', matrix.T)
print('\nnp.transpose:\n', np.transpose (matrix))
```

    
    matrix:
     [[1 2 3]
     [4 5 6]
     [7 8 9]]
    
    matrix.T:
     [[1 4 7]
     [2 5 8]
     [3 6 9]]
    
    np.transpose:
     [[1 4 7]
     [2 5 8]
     [3 6 9]]
    

np.**linalg**.<font color = magenta><b>det</b></font> (ndarray) and np.**linalg**.<font color = magenta><b>matrix_rank</b></font>


```python
def is_det_zero (det):
    return np.allclose ([0], [det])

print('\nmatrix:\n', matrix)
det = np.linalg.det(matrix)
print('\nnp.linalg.det:\n', det)
print ('\nis determinant zero?:\n', is_det_zero (det))
rank = np.linalg.matrix_rank(matrix)
print('\nnp.linalg.matrix_rank:\n', rank)
```

    
    matrix:
     [[1 2 3]
     [4 5 6]
     [7 8 9]]
    
    np.linalg.det:
     -9.51619735392994e-16
    
    is determinant zero?:
     True
    
    np.linalg.matrix_rank:
     2
    

ndarray.<font color = magenta><b>diagonal</b></font>


```python
print('\nmatrix:\n', matrix)
print('\nThe Principal diagonal:\n', matrix.diagonal())
print('\nThe diagonal at offset 1:\n', matrix.diagonal(offset=1))
print('\nThe diagonal at offset -1:\n', matrix.diagonal(offset=-1))
print('\nThe diagonal at offset -2:\n', matrix.diagonal(offset=-2))
```

    
    matrix:
     [[1 2 3]
     [4 5 6]
     [7 8 9]]
    
    The Principal diagonal:
     [1 5 9]
    
    The diagonal at offset 1:
     [2 6]
    
    The diagonal at offset -1:
     [4 8]
    
    The diagonal at offset -2:
     [7]
    

ndarray.<font color = magenta><b>trace</b></font>

The trace of a matrix is the **sum of its diagonal components**.


```python
print('\nmatrix:\n', matrix)
trace = matrix.trace ()
print ('\ntrace:\n', trace)
```

    
    matrix:
     [[1 2 3]
     [4 5 6]
     [7 8 9]]
    
    trace:
     15
    

ndarray.<font color = magenta><b>eig</b></font> - eigenvalues and eigenvectors


```python
print('\nmatrix:\n', matrix)
evalues, evectors = np.linalg.eig (matrix)
print ('\nEigenvalues:\n', evalues)
print ('\nEigenvectors:\n', evectors)
```

    
    matrix:
     [[1 2 3]
     [4 5 6]
     [7 8 9]]
    
    Eigenvalues:
     [ 1.61168440e+01 -1.11684397e+00 -9.75918483e-16]
    
    Eigenvectors:
     [[-0.23197069 -0.78583024  0.40824829]
     [-0.52532209 -0.08675134 -0.81649658]
     [-0.8186735   0.61232756  0.40824829]]
    

np.**linalg**.<font color = magenta><b>inv</b></font> - inverse of a matrix


```python
print('\nmatrix:\n', matrix)
inverse = np.linalg.inv (matrix)
print ('\ninverse:\n', inverse)
```

    
    matrix:
     [[1 2 3]
     [4 5 6]
     [7 8 9]]
    
    inverse:
     [[ 3.15251974e+15 -6.30503948e+15  3.15251974e+15]
     [-6.30503948e+15  1.26100790e+16 -6.30503948e+15]
     [ 3.15251974e+15 -6.30503948e+15  3.15251974e+15]]
    

solving <font color = magenta><b>systems of linear equations</b></font> - Ax = b : solve for x


```python
A = np.array([[2,1,-2],[3,0,1],[1,1,-1]])
b = np.transpose(np.array([[-3,5,-2]]))
print('\nA:\n', A)
print('\nb:\n', b)
x = np.linalg.solve (A, b)
print ('\nAx = b. x =:\n', x)
```

    
    A:
     [[ 2  1 -2]
     [ 3  0  1]
     [ 1  1 -1]]
    
    b:
     [[-3]
     [ 5]
     [-2]]
    
    Ax = b. x =:
     [[ 1.]
     [-1.]
     [ 2.]]
    

# H. <font color = orange><b>Random Sampling from Distributions</b></font>

**TODO**
Note:
- randn generates samples from the normal distribution, while numpy. random. **rand from uniform** (in range [0,1)).
- np.cumsum (np.random.randn(10,1)) - plot

## i. np.random.<font color = magenta><b>randn</b></font> and np.random.<font color = magenta><b>standard_normal</b></font>

**Specific** Normal Distribution
- <font color = magenta><b>mean = 0</b></font>
- <font color = magenta><b>variance = 1</b></font>

### randn

> backward compatibility with **Matlab**
> - takes dimensions as individual parameters

randn **(n)** returns a **1D** ndarray


```python
randn (3)
```




    array([-0.69781929,  0.76250905, -0.00100293])



randn **(m, n)** returns a **2D** (m x n) ndarray


```python
randn (3, 2)
```




    array([[ 0.01257397, -1.09267948],
           [ 0.43461778, -0.9010176 ],
           [ 0.94876503, -0.06544939]])



randn **(i, j, k)** returns a **3D** (i x j x k) ndarray


```python
randn (2, 3, 4)
```




    array([[[-0.18420962, -2.80150569, -1.94776301,  0.58616938],
            [-0.16765517,  0.0843139 ,  0.88771571,  0.05693744],
            [ 0.88764414, -0.92584994,  0.96424221,  2.3480603 ]],
    
           [[-0.14195735, -0.03706071,  0.19416724, -1.05178575],
            [-0.66259882, -1.4020511 ,  0.87980418, -0.7594163 ],
            [-0.07895493,  0.68616642, -1.58868401,  1.62971673]]])



**plot**


```python
sns.kdeplot (randn (500))
plt.show ()
```


![png]({{site.baseurl}}/assets/images/numpy-playground/output_268_0.png)


### standard_normal

> **NumPy-centric**
> - takes dimensions as a tuple
> - This allows other parameters like **dtype** and **order** to be passed to the function as well.

standard_normal (**(m, n)**) returns a **2D** (m x n) ndarray


```python
standard_normal ((2, 4))
```




    array([[-1.86646271, -0.19924265,  1.13467334, -0.17763385],
           [-0.21805904, -0.47804114,  0.5908614 ,  1.49768637]])



**plot**


```python
sns.kdeplot (standard_normal (500))
plt.show ()
```


![png]({{site.baseurl}}/assets/images/numpy-playground/output_274_0.png)


## ii. np.random.<font color = magenta><b>normal</b></font>

**Generic** Normal Distribution
- mean: <font color = magenta><b>loc = </b></font>
- variance: <font color = magenta><b>scale =</b></font>

> **NumPy-centric**
> - takes dimensions as a tuple
> - This allows other parameters like **loc** and **scale** to be passed to the function as well.

normal (**loc** = , **scale** = , **(m, n)**) returns a **2D** (m x n) ndarray with values having 'loc' as mean, and 'scale' as variance


```python
normal (loc=15.0, scale=5.0, size=(2,3))
```




    array([[ 7.8948123 ,  4.39815973,  9.39660432],
           [20.40527397, 19.64774283,  9.2247971 ]])



**plot**


```python
sns.kdeplot (normal (loc = 15.0, scale = 5.0, size = 500))
plt.show ()
```


![png]({{site.baseurl}}/assets/images/numpy-playground/output_281_0.png)


## iii. np.random.<font color = magenta><b>seed</b></font> and np.random.<font color = magenta><b>RandomState</b></font>

https://stackoverflow.com/questions/5836335/consistently-create-same-random-numpy-array/5837352#5837352


```python

```
