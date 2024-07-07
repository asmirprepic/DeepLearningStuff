import tensorflow as tf

"""
Some linear algebra snippet examples in TF
"""

## Scalar Examples

x = tf.constant(2.0) # setting a constatn
y = tf.constant(3.0) # setting a constant

x+y,x*y,x/y,x**y # some mathematical operatins

## Vector examples
x = tf.range(3) # setting a vector
x[2] # indexing a vector

len(x) # getting the lenght of the vector


## Matrix examples
A = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3)) # setting a 2x3 matrix in tf
tf.transpose(A) # transposing the vector in tf

## Tensor examples
tf.reshape(tf.range(24),(2,3,4)) # setting a 2x3x4  tensor

A =tf.reshape(tf.range(6,dtype = tf.float32),(2,3))
B = A 
A, A+B

# Hadamard product
A*B

# Adding a scalar
a = 2
a + X

# Reduction (This is the sum of the elements of the tensor)
x = tf.range(3,dtype = tf.float32)
x,tf.reduce_sum(x)


## Getting the mean 
tf.reduce_mean(A),tf.reduce_sum(A)/tf.size(A).numpy() ## Produces the same results. 

## Dot product
y = tf.ones(3,dtype = tf.float32)
x,y =  tf.tensordot(x,y,axes=1) ## Should retunr 3 since its the dot procut of [0,1,2] and [1,1,1]


## Matrix vector operations
A.shape, x.shape, tf.linalg.matvec(A,x) ## should be a 2 dimensional vector


## Matrix matrix operations
B = tf.ones((3,4),tf.float32)
tf.matmul(A,B)


# Norms 
u = tf.constant([3.0,-4.0])
tf.norm(u)


# L1 norm
tf.reduce_sum(tf.abs(u))
















