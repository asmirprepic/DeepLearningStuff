import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization  # Import necessary Keras layers
from tensorflow.keras import Sequential 
from deep_learning.DeepLearningStuff.Classes.blocks_layers.multihead_attention_layer import MultiHeadAttention

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self,units,num_heads,ff_dim):
        super(TransformerBlock,self).__init__()
        self.attention = MultiHeadAttention(num_heads,units)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim,activation = 'relu'),
            Dense(units)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)

    def call(self,inputs):
        attn_output = self.attention(inputs)
        attn_output = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(attn_output)
        return self.layernorm2(attn_output+ ffn_output)