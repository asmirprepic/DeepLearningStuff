import tensorflow as tf
from tensorflow.keras.layers import Dense,LayerNormalization,Add
from multihead_attention_layer import MultiHeadAttention

class BottleNeckTransformerBlock(tf.keras.layers.Layer):
    def __init__(self,embed_dim,num_heads,bottleneck_dim):
        super(BottleNeckTransformerBlock,self).__init__()
        self.attention = MultiHeadAttention(num_heads,embed_dim = embed_dim)
        self.bottleneck = Dense(bottleneck_dim,activation = 'relu')
        self.layernorm1 = LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = LayerNormalization(epsilon = 1e-6)

    def call(self,inputs):
        # self attention
        attn_output = self.attention([inputs,inputs,inputs])
        attn_output = self.layernorm1(inputs + attn_output)
        
        # Bottleneck layer
        bottleneck_output = self.bottleneck(attn_output)
        return self.layernorm2(attn_output + bottleneck_output)