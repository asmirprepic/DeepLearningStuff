import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
  """
  Implements positional encoding for the Transformer model.
  This layer generates a matrix of sine and cosine functions to represent positional information 
  for each token in a sequence. The positional encodings are added to the input embeddings to 
  provide the model with information about the relative and absolute positions of tokens in the sequence.
  """
  def __init__(self,sequence_length,embed_dim):
    """
    Initializes the PositionalEncoding layer.

    Args:
    ----------
        sequence_length (int): The length of the input sequences (i.e., number of tokens in each sequence).
        embed_dim (int): The dimensionality of the embedding vectors (i.e., the number of features in each token embedding).
    """
    super(PositionalEncoding,self).__init__()
    self.pos_encoding = self.positional_encoding(sequence_length,embed_dim)

  def get_angles(self,pos,i,embed_dim):
    """
    Calculates the angles for the sine and cosine positional encodings.

    Args:
    ------------
        pos (Tensor): A tensor representing the positions of the tokens in the sequence (shape: [sequence_length, 1]).
        i (Tensor): A tensor representing the embedding dimension indices (shape: [1, embed_dim]).
        embed_dim (int): The dimensionality of the embeddings.

    Returns:
    ----------
        Tensor: A tensor of angles used for computing sine and cosine values (shape: [sequence_length, embed_dim]).
    """
    angle_rates = 1 /tf.pow(10000,(2*(i//2)) / tf.cast(embed_dim,tf.float32))
    return pos*angle_rates

  def positional_encoding(self,sequence_length,embed_dim):
    """
        Generates the full positional encoding matrix.

        Args:
            sequence_length (int): The length of the input sequences.
            embed_dim (int): The dimensionality of the embeddings.

        Returns:
            Tensor: A tensor representing the positional encodings (shape: [1, sequence_length, embed_dim]).
        """
    angle_rads = self.get_angles(
      pos = tf.range(sequence_length)[:,tf.newaxis],
      i = tf.range(embed_dim)[tf.newaxis,:],
      embed_dim= embed_dim)

    sines = tf.math.sin(angle_rads[:,0::2])
    cosines =tf.math.cos(angle_rads[:,1::2])

    pos_encoding = tf.concat([sines,cosines],axis = -1)
    pos_encoding = pos_encoding[tf.newaxis,...]

    return tf.cast(pos_encoding,tf.float32)

  def call(self,inputs):
    """
    Adds positional encoding to the input embeddings.

    Args:
        inputs (Tensor): A tensor of input embeddings (shape: [batch_size, sequence_length, embed_dim]).

    Returns:
        Tensor: The input embeddings with added positional encodings (same shape as inputs).
    """
    return inputs + self.pos_encoding[:,:tf.shape(inputs)[1],:]
    
    
