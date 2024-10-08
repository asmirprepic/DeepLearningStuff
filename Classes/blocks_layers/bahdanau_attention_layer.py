import tensorflow as tf
from tensorflow.keras.layers import Dense

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self,units):
        super(BahdanauAttention,self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self,query,values):
        query_with_time_axis =tf.expand_dims(query,1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        attention_weights = tf.nn.softmax(score,axis = 1)
        context_vector = attention_weights*values
        context_vector = tf.reduce_sum(context_vector,axis =1)
        return context_vector,attention_weights