import tensorflow as tf
from tensorflow.keras import layers,models

class SELayer(layers.Layer):
    def __init__(self,reduction = 16, **kwargs):
        super(SELayer,self).__init__(**kwargs)
        self.reduction = reduction
    
    def build(self,input_shape):
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(input_shape[-1]//self.redcution,activation = 'relu')
        self.dense2 = layers.Dense(input_shape[-1],activation = 'sigmoid')
        super(SELayer,self).build(input_shape)

    def call(self,inputs):
        se = self.global_avg_pool(inputs)
        se = self.dense1(se)
        se = self.dense2(se)
        se = tf.expand_dims(se,axis = 1)
        se = tf.expand_dims(se,axis = 1)
        return inputs*se