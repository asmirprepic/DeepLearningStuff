import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class Time2Vec(Layer):
    def __init__(self,kernel_size):
        super(Time2Vec,self).__init__()
        self.dense_linear = Dense(1,use_bias = True)
        self.dense_periodic = Dense(kernel_size-1,use_bias = False)
    
    def call(self,inputs):
        time_linear = self.dense_linear(inputs)
        time_periodic = tf.sin(self.dense_periodic(inputs))
        return tf.concat([time_linear,time_periodic],-1)