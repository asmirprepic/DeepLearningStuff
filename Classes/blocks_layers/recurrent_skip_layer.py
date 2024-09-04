import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense


class RecurrentSkip(Layer):
    """
    Recurrent skip layer for capturing long-term dependencies by skipping time steps. 

    This layer implements skip connections in a recurrent network, allowing the model 
    to skip over time steps and focus on long-term relationships in the data. 


    Attributes: 
    ---------------
    units (int):
        Number of units in the Dense Layer applied to skipped time steps. 
    skip_steps (int):
        Number of time steps to skip during each forward pass.     
    """

    def __init__(self,units,skip_steps):
        super(RecurrentSkip,self).__init__()
        self.units = units
        self.skip_steps = skip_steps
        self.dense = Dense(units)

    def call(self,inputs):  
        """
        Forward pass of the recurrent skip layer. 

        Arguments: 
        ------------------
        inputs (Tensor):
            Input time series data in the shape of (batch_size, sequence_lenght, features).

        Returns: 
        --------------
        Tensor:
            The output after skipping and applying a dense layer
    
        """
        outputs = []
        for t in range(0,inputs.shape[1],self.skip_steps):
            output = self.dense(inputs[:,t,:])
            outputs.append(output)
        outputs = tf.stack(outputs,axis = 1)

        return outputs