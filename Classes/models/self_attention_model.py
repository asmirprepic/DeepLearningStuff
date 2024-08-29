import tensorflow as tf
from tensorflow.keras import layers,models
from deep_learning.DeepLearningStuff.Classes.blocks_layers.self_attention_layer import SelfAttention

class SelfAttionModel(tf.keras.Model):
    def __init__(self,num_classes,units):
        super(SelfAttionModel,self).__init__()

        # Define layers
        self.conv1 = layers.Conv1D(64,kernel_size = 3,padding = 'same',activation = 'relu')
        self.conv2 = layers.Conv1D(64,kernel_size = 3,padding = 'same',activation = 'relu')
        self.attention = SelfAttention(units)
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense = layers.Dense(256,activation = 'relu')
        self.output_layer = layers.Dense(num_classes,activation = 'softmax')

    def call(self,inputs,training = False,mask = None):
        # Forward pass through the layers
        x = self.conv1(inputs)
        x = self.conv2(x)
        attention_output, attentinon_weights = self.attention(x)
        x = self.global_pool(attention_output)
        x = self.dense(x)
        return self.output_layer(x)
    
    