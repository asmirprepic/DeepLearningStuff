import tensorflow as tf
from tensorflow.keras.layer import Dense,Layer


class CustomGRU(tf.keras.layers.Layer):

    """
    Custom implementation of Gated Recurrent Unit (GRU) cell. 

    This GRU is a simplified version of GRU used in recurrent neural networks. 
    
    Attributs: 
    -------------
    units, (int):
        Number of units (hidden state size) in the GRU cell
    """

    def __init__(self,units):
        """
        Initalizes the CustomGRU cell with specified number of units. 

        Parameters: 
        --------------
        units : int
            The number of units (neurons) in the GRU cell.        
        """

        super(CustomGRU,self).__init__()
        self.units = units
    
    def build(self,input_shape):
        """
        Creates the trainable cell for the GRU cell. 

        This method is called automatically by Tensorflow the first time the layer is used. 
        It initializes the weights for the update gate (z), reset gate (r), and candidate hidden state (h_hat).

        Parameters: 
        ---------------
        input shape: tuple
            The shape of the input tensor, used to determine the dimensions of the weight matrices
        """

        # Update the weights
        self.Wz = self.add_weight(shape=(input_shape[-1],self.units),
                                  initializer = 'glorot_uniform',
                                  trainable = 'True')
        self.Uz = self.add_weight(shape=(self.units,self.units),
                                  initializer = 'glorot_uniform',
                                  trainable = 'True')
        self.bz = self.add_weight(shape=(self.units,self,),
                                  initializer = 'zeros',
                                  trainable = True)
        
        # Reset gate weights
        self.Wr  = self.add_weight(shape =(input_shape[-1],self.units),
                                    initializer = 'glorot_uniform',
                                    trainable = True)
        self.Ur = self.add_weight(shape = (self.units,self.units),
                                  initalizer = 'glorot_uniform',
                                  trainable = True)
        self.br = self.add_weight(shape=(self.units,self,),
                                  initializer = 'zeros',
                                  trainable = True)
        
        # Candidate hidden state weights
        self.Wr  = self.add_weight(shape =(input_shape[-1],self.units),
                                    initializer = 'glorot_uniform',
                                    trainable = True)
        self.Ur = self.add_weight(shape = (self.units,self.units),
                                  initalizer = 'glorot_uniform',
                                  trainable = True)
        self.br = self.add_weight(shape=(self.units,self,),
                                  initializer = 'zeros',
                                  trainable = True)
        

    def call(self,inputs,states):
        """
        Performs the forward pass for the GRU cell. 
        Computes the new hidden state for the current time step using the update and reset gates. 
        The hidden state is updated based on the previous hidden state and the current input.

        Parameters: 
        -----------------
        inputs : Tensor
            The input tensor at the current time step (shape: [batch_size,input_dim])
        states: list of Tensors
            The previous hidden states (shape: [batch_size,units])
        
        Returns: 
        ----------------
        h: Tensor
            The new hidden state for the current time step
        [h]: list of Tensors
            A list containing the new hidden state (useful for RNN layers)

        """

        h_prev = states[0]

        # Update gate: determines how much of the previous hidden state should be carried forward.
        z = tf.nn.sigmoid(tf.matmul(inputs,self.Wz) + tf.matmul(h_prev,self.Uz) + self.bz)

        # Reset gate: determines how much of the previous hidden state should be ignored.
        r = tf.nn.sigmoid(tf.matmul(inputs,self.Wz) + tf.matmul(h_prev,self.Ur) + self.br)

        # Candidate hidden state: the potential new hidden state, condidering the reset gate. 
        h_hat = tf.nn.tanh(tf.matmul(inputs,self.Wh) + tf.matmul(r*h_prev,self.Uh) + self.bh)

        # New hidden state: a combination of the previous hidden state and the candiate hidden state, 
        # weighted by the update gate
        h = z*h_prev + (1-z)*h_hat

        return h,[h]