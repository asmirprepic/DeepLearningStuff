"""
illustration of the chain rule using tensorflow

"""


import tensorflow as tf

x = tf.constant(1.)
w1 = tf.Variable(2.)
b1 = tf.Variable(1.)
w2 = tf.Variable(2.)
b2 = tf.Variable(1.)

# Create a persistent gradient recorder
with tf.GradientTape(persistent=True) as tape:
    # Create a two-layer neural network
    y1 = x * w1 + b1
    y2 = y1 * w2 + b2

# Solve partial derivatives
dy2_dy1 = tape.gradient(y2, y1)
dy1_dw1 = tape.gradient(y1, w1)
dy2_dw1 = tape.gradient(y2, w1)

# Validate the chain rule
print("dy2_dy1 * dy1_dw1:", dy2_dy1.numpy() * dy1_dw1.numpy())
print("dy2_dw1:", dy2_dw1.numpy())

# Release resources held by the persistent tape
del tape
