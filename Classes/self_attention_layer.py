class SelfAttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_q = self.add_weight(name='query_weight', shape=(input_shape[-1], input_shape[-1]), initializer='random_normal', trainable=True)
        self.W_k = self.add_weight(name='key_weight', shape=(input_shape[-1], input_shape[-1]), initializer='random_normal', trainable=True)
        self.W_v = self.add_weight(name='value_weight', shape=(input_shape[-1], input_shape[-1]), initializer='random_normal', trainable=True)
        super(SelfAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        Q = tf.keras.backend.dot(inputs, self.W_q)
        K = tf.keras.backend.dot(inputs, self.W_k)
        V = tf.keras.backend.dot(inputs, self.W_v)

        attention_weights = tf.keras.backend.softmax(tf.keras.backend.batch_dot(Q, K, axes=[2, 2]) / tf.keras.backend.sqrt(tf.keras.backend.cast(tf.keras.backend.shape(K)[-1], tf.keras.backend.floatx())))
        context = tf.keras.backend.batch_dot(attention_weights, V)

        return context
