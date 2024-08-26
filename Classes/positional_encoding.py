import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
  def __init__(self,sequence_lenght,embed_dim):
    super(PositionalEncoding,self).__init__():
    self.pos_encoding = self.positional_encoding(sequence_length,embed_dim)

  def get_angles(self,pos,i,embed_dim):
    angle_rates = 1 /tf.pow(10000,(2*(i//2)) / tf.cast(embed_dim,tf.float32))
    return pos*angle_rates

  def positional_encoding(self,sequence_lenght,embed_dim):
    angle_rads = self.get_angles(
      pos = tf.range(sequence_length)[:,tf.newaxis],
      i = tf.range(embed_dim)[tf.newaxis,:],
      embed_dim)

    sins = tf.math.sin(angle_rads[:,0::2])
    cosines =tf.math.cos(angle_rads[:,1::2])

    pos_encoding = tf.concat([sines,cosines],axis = -1)
    pos_encoding = pos_encoding[tf.newaxis,...]

    return tf.cast(pos_encoding,tf.float32)

  def call(self,inputs):
    return inputs + self.pos_encoding[:,:tf.shape(inputs)[1],:]
    
    
