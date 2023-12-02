"""
Illustrating the automatic differentiation in TF
"""

x = tf.range(4,dtype = tf.float32) ## setting a vector with values

x = tf.Varaible(x)  ## Setting a value for a shared memory. Variable can be manipulated. 

with tf.GradientTape(x) as t: 
  y = 2*tf.tensordot(x,x,axes=1)   # Recording all computations for a function and store in y

x_grad = t.gradient(x,y) 

x_grad = 4*x #Since y = 2xTx should have gradient 4x this just checks it



  
