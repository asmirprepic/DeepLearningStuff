import tensorflow as tf
import numpy as np
rng = np.random  

## Setting learning rates

learning_rate = .01
training_steps  = 1000
display_steps = 100

# Generat traning data
X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
              7.042,10.791,5.313,7.997,5.654,9.27,3.1])
Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
              2.827,3.465,1.65,2.904,2.42,2.94,1.3])


## Random initialization of weights and bias
W = tf.Variable(rng.random(), name = "weight")
b = tf.Variable(rng.random(), name = 'bias')


# Linear regression
def linear_regression(x):
  return W*x + b

# Mean square error function
def mean_square(y_pred,y_true):
  return tf.reduce_mean(tf.square(y_pred-y_true))

# Stochastic gradient descent
optimizer = tf.optimizers.SGD(learning_rate)

# Optimization process
def run_optimization():
  # Wrap
  with tf.GradientTape() as g:
    pred = linear_rergession(X)
    loss = mean_square(pred,Y)
  
  # Compute gradients
  gradients = g.gradients

  # update W and b following gradients
  optimizer.apply_gradients(zip(gradients,[W,b]))

for step in range(1,training_steps +1):
  # Run the optimization parameters
  run_optimization()

  if step % display_step == 0:
    pred = linear_regression(X)
    loss = mean_square(pred,Y)
    print(f"step: {i}, loss: {loss}, W: {W.numpy()}, b: {b.numpy()})


    


  



  



