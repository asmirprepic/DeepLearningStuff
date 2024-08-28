import tensorflow as tf
from tensorflow.keras import layers

class SELayer1D(layers.Layers):
    def __init__(self,reduction = 16, **kwargs):
        self.reduction = reduction 

    def build(self,input_shape):
        # Initialize layers based on the input shape
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(input_shape[-1] // self.reduction,activation = 'relu')
        self.dense2 = layers.Dense(input_shape[-1],activation = 'sigmoid')
        super(SELayer1D,self).build(input_shape)
    
    def call(self,inputs):
        # Squeeze ste
        se = self.global_avg_pool(inputs)
        se = self.dense1(se)
        se = self.dense2(se)

        # Reshape for broadcasting
        se = tf.expand_dims(se,axis = 1)

        return inputs*se