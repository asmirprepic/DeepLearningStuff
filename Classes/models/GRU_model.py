import tensorflow as tf
from tensorflow.keras import layers, models

class GRUModel(tf.keras.Model):
    def __init__(self, units, num_outputs):
        super(GRUModel, self).__init__()
        # Define layers
        self.gru1 = layers.GRU(units, return_sequences=True)
        self.gru2 = layers.GRU(units)
        self.dense = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(num_outputs)

    def call(self, inputs, training=False):
        x = self.gru1(inputs)
        x = self.gru2(x)
        x = self.dense(x)
        return self.output_layer(x)

# Example usage
units = 128
num_outputs = 1# For single-output regression

model = GRUModel(units=units, num_outputs=num_outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.build(input_shape=(None, 100, 64))
model.summary()
