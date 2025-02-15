import tensorflow as tf
import numpy as np
import math

class PositionalEncoding(tf.keras.layers.Layer):
    """
    Implements positional encoding as described in "Attention is All You Need".
    This layer adds positional information to token embeddings.
    
    Attributes:
        max_seq_len (int): Maximum sequence length for which the encoding is computed.
        d_model (int): Dimensionality of the embeddings.
        dropout_rate (float): Dropout rate applied after adding positional encoding.
        pos_encoding (tf.Tensor): Precomputed positional encoding matrix.
    """
    
    def __init__(self, max_seq_len: int, d_model: int, dropout_rate: float = 0.1):
        """
        Initializes the PositionalEncoding layer.
        
        Args:
            max_seq_len (int): Maximum sequence length.
            d_model (int): Embedding dimensionality.
            dropout_rate (float): Dropout rate applied after adding positional encoding.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        
        self.pos_encoding = self._compute_positional_encoding(max_seq_len, d_model)

    def _compute_positional_encoding(self, max_seq_len: int, d_model: int) -> tf.Tensor:
        """
        Computes the positional encoding matrix.
        
        Args:
            max_seq_len (int): Maximum sequence length.
            d_model (int): Embedding dimensionality.
        
        Returns:
            tf.Tensor: A tensor of shape (1, max_seq_len, d_model) containing the positional encodings.
        """
        
        pos = np.arange(max_seq_len)[:, np.newaxis]
        
        i = np.arange(d_model)[np.newaxis, :]
        
        
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

        
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        
        pos_encoding = angle_rads[np.newaxis, ...]  
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Adds positional encoding to the input embeddings.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            training (bool): Whether the layer should behave in training mode.
        
        Returns:
            tf.Tensor: Tensor with positional encoding applied, shape (batch_size, seq_len, d_model).
        """
        seq_len = tf.shape(inputs)[1]
        
        outputs = inputs + self.pos_encoding[:, :seq_len, :]
        return self.dropout(outputs, training=training)
