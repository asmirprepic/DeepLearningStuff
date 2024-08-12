import tensorflow as tf
from tensorflow.keras.layers import Dense

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.units = units
        self.query_dense = Dense(units)
        self.key_dense = Dense(units)
        self.value_dense = Dense(units)

    def call(self, inputs):
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Scaled dot-product attention
        score = tf.matmul(query, key, transpose_b=True)
        score = score / tf.math.sqrt(tf.cast(self.units, tf.float32))
        weights = tf.nn.softmax(score, axis=-1)

        output = tf.matmul(weights, value)
        return output, weights