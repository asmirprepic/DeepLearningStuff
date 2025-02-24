class RealNVPCouplingNN(tf.keras.Model):
    def __init__(self, output_units):
        """
        A simple neural network that outputs shift and log-scale parameters.
        Args:
            output_units (int): Number of output units (for a 2D input with num_masked=1, output_units should be 1).
        """
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense_out = tf.keras.layers.Dense(2 * output_units, activation=None)
        
    def call(self, x):
        h = self.dense1(x)
        h = self.dense2(h)
        out = self.dense_out(h)
        shift, log_scale = tf.split(out, num_or_size_splits=2, axis=-1)
        return shift, log_scale
