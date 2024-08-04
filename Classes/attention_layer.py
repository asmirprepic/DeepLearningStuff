import tensorflow as tf
from tensorflow.keras import layers, models


class AttentionLayer(layers.Layer):
    def __init__(self,**kwargs):
        super(AttentionLayer,self).__init__(**kwargs)
    
    def build(self,input_shape):
        self.W = self.add_weight(name = 'attention_weight',shape = (input_shape[-1],1),initializer = 'random_normal',trainable = True)
        self.b = self.add_weight(name = 'attention_bias',shape = (input_shape[1],1),initializer = 'zeros',trainable = True)
        super(AttentionLayer,self).build(input_shape)
    
    def call(self,inputs):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs,self.W)+ self.b)
        e = tf.keras.backend.squeeze(e,axis = -1)
        alpha = tf.keras.backend.softmax(e)
        alpha = tf.keras.backend.expand_dims(alpha,axis = -1)
        context = inputs*alpha
        context = tf.keras.backend.sum(context,axis = 1)
        return context
