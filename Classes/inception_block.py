import tensorflow as tf
from tensorflow.keras import layers

class InceptionBlock(layers.Layer):
    def __init__(self,filters,**kwargs):
        super(InceptionBlock).__init__(**kwargs)
        self.conv1x1 = layers.Conv1D(filters,1, padding = 'same',activation = 'relu')
        self.conv3x3 = layers.Conv1D(filters,3,padding = 'same',activation = 'relu' )
        self.conv5x5 = layers.Conv1D(filters,5,padding = 'same',activation = 'relu')
        self.concat =layers.Concatenate()

    def call(self,inputs):
        x1 = self.conv1x1(inputs)
        x2 = self.conv3x3(inputs)
        x3 = self.conv5x5(inputs)

        return self.concat([x1,x2,x3])