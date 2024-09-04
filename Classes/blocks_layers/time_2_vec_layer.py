import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class Time2Vec(Layer):
    """
    Time2Vec layer for capturing periodic patterns in time series data. 
    Generalizes the concept of positional encoding to time series by 
    learning time representations through linear and periodic functions. 

    Attributes: 
    ------------
    kernel_size (int): Number of time-embedding dimensions. 

    """
    def __init__(self,kernel_size):
        super(Time2Vec,self).__init__()
        self.dense_linear = Dense(1,use_bias = True)
        self.dense_periodic = Dense(kernel_size-1,use_bias = False)
    
    def call(self,inputs):
        """
        Forawrd pass of Time2VecLayer

        Parameters: 
        ---------------
        inputs: Tensor
            Input time series data of shape (batch_size, sequence_length,1)
        
        Returns: 
        ---------------
        Tensor
            Time2Vec embedding with both linear and periodic components. 
    
        """
        time_linear = self.dense_linear(inputs)
        time_periodic = tf.sin(self.dense_periodic(inputs))
        return tf.concat([time_linear,time_periodic],-1)