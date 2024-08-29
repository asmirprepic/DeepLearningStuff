
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization  # Import necessary Keras layers
from tensorflow.keras import Sequential 
from deep_learning.DeepLearningStuff.Classes.blocks_layers.self_attention_layer import SelfAttention

class MultiHeadAttention(tf.kears.layers.Layer):
    def __init__(self,num_heads,units):
        super(MultiHeadAttention,self).__init__()
        self.num_heads = num_heads
        self.units = units

        self.head_dim = units/num_heads
        self.query_dense = Dense(units)
        self.key_dense = Dense(units)
        self.value_dense = Dense(units)
        self.dense = Dense(units)

    def split_heads(self,x,batch_size):
        x = tf.reshape(x,(batch_size,-1,self.num_heads,self.head_dim))
        return tf.transpose(x, perm = [0,2,1,3])
    
    def call(self,inputs):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value= self.value_dense(inputs)

        query = self.split_heads(query,batch_size)
        key = self.split_heads(key,batch_size)
        value = self.split_heads(value,batch_size)

        attention, _ = SelfAttention(self.head_dim)(query,key,value)
        attention = tf.transpose(attention,perm= [0,2,1,3])
        concat_attention = tf.reshape(attention,(batch_size,-1,self.units))

        output = self.dense(concat_attention)

        return output