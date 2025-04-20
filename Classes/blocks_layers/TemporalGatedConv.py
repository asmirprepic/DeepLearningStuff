class TemporalGatedConv(tf.keras.layers.Layer):
    """
    Temporal Gated Convolution Layer. Combines 1D convolution with a gating mechanism.
    """
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.conv_feature = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')
        self.conv_gate = tf.keras.layers.Conv1D(filters, kernel_size, padding='same', activation='sigmoid')

    def call(self, inputs):
        feature = self.conv_feature(inputs)
        gate = self.conv_gate(inputs)
        return feature * gate  # Element-wise gating
