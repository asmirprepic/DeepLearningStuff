import tensorflow as tf
from tensorflow.keras import layers

class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='same')
        self.activation = layers.Activation('relu')
        self.add = layers.Add()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.add([x, inputs])
        return self.activation(x)
