
import tensorflor as tf
from tensorflow.keras import layers

class HighwayLayer(layers.Layer,**kwargs):
    def __init__(self,units,**kwargs):
        super(HighwayLayer,self).__init__(**kwargs)
        self.units = units

    def build(self,input_shape):
        self.dense_H = layers.Dense(self.units)
        self.dense_T = layers.Dense(self.units,activation = 'sigmoid',bias_initializer = 'ones')
        super(HighwayLayer.self).build(input_shape)
    
    def call(self,inputs):
        H = self.dense_H(inputs)
        T = self.dense_T(inputs)
        C = 1-T
        return H*T+inputs*C
    
    