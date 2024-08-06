import tensorflow as tf
from tensorflow.keras import layers

class LayerNormalization(layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self,input_shape):
        self.gamma = self.add_weight(name = 'gamma',shape = input_shape[-1:],initializer = 'ones',Trainable = True)
        self.beta = self.add_weight(name = 'beta',shape = input_shape[-1:],initializer = 'zeros',Trainable = True)
        super(LayerNormalization,self).build(input_shape)

    def call(self,inputs):
        mean = tf.keras.backend.mean(inputs,axis = -1,keepdims = True)
        variance = tf.keras.backend.var(inputs,axis = -1,keepdims = True)
        normalized_inputs = (inputs-mean)/tf.keras.backend.sqrt(variance + self.epsilon)
        return self.gamma*normalized_inputs + self.beta