import tensorflow as tf
from tensorflow.keras import layers

class CustomGRUCell(layers.Layer):
    def __init__(self, units):
        super(CustomGRUCell, self).__init__()
        self.units = units

    def build(self, input_shape):
        # Define trainable weights for update gate, reset gate, and candidate state
        self.Wz = self.add_weight(shape=(input_shape[-1], self.units),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Uz = self.add_weight(shape=(self.units, self.units),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.bz = self.add_weight(shape=(self.units,),
                                  initializer='zeros',
                                  trainable=True)
        
        self.Wr = self.add_weight(shape=(input_shape[-1], self.units),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Ur = self.add_weight(shape=(self.units, self.units),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.br = self.add_weight(shape=(self.units,),
                                  initializer='zeros',
                                  trainable=True)

        self.Wh = self.add_weight(shape=(input_shape[-1], self.units),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Uh = self.add_weight(shape=(self.units, self.units),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.bh = self.add_weight(shape=(self.units,),
                                  initializer='zeros',
                                  trainable=True)

    def call(self, inputs, states):
        h_prev = states[0]

        # Update gate
        z = tf.nn.sigmoid(tf.matmul(inputs, self.Wz) + tf.matmul(h_prev, self.Uz) + self.bz)
        
        # Reset gate
        r = tf.nn.sigmoid(tf.matmul(inputs, self.Wr) + tf.matmul(h_prev, self.Ur) + self.br)
        
        # Candidate hidden state
        h_hat = tf.nn.tanh(tf.matmul(inputs, self.Wh) + tf.matmul(r * h_prev, self.Uh) + self.bh)
        
        # New hidden state
        h = z * h_prev + (1 - z) * h_hat

        return h, [h]

class CustomGRU(layers.Layer):
    def __init__(self, units, return_sequences=False, return_state=False):
        super(CustomGRU, self).__init__()
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.gru_cell = CustomGRUCell(units)

    def call(self, inputs):
        # Initialize the hidden state as a tensor of zeros
        batch_size = tf.shape(inputs)[0]
        h = tf.zeros((batch_size, self.units))
        
        # Iterate over each time step
        outputs = []
        for t in range(inputs.shape[1]):
            h, _ = self.gru_cell(inputs[:, t, :], [h])
            if self.return_sequences:
                outputs.append(h)
        
        if self.return_sequences:
            outputs = tf.stack(outputs, axis=1)
        else:
            outputs = h
        
        if self.return_state:
            return outputs, h
        else:
            return outputs

# Example usage of the custom GRU layer
units = 128
num_outputs = 1# For single-output regression

inputs = tf.random.normal((32, 100, 64))  # Example input: (batch_size, time_steps, features)
custom_gru_layer = CustomGRU(units=units, return_sequences=True)
outputs = custom_gru_layer(inputs)
print(outputs.shape)  # (32, 100, 128)