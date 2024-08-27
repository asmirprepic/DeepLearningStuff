import tensorflow as tf
from tensorflow.keras import Sequential
from tensflow.keras.layers import Dense, Input,TimeDistrubuted
from custom_gru_block import CustomGRU


# Define the model class

class CustomGRUModel(tf.keras.Model):
    """
    A sequence model built using the CustomGRU class

    The model takes a sequence of input data, process it through a GRU cell, and ouputs the predictins
    of the next value in the sequence

    Attributes: 
    ----------------
    units: int
        Number of units in the CustomGRU block.
    output_dim:
        Number of output units (e.g. 1 for regression)

    """

    def __init__(self,units,output_dim):
        """
        Initiaizes the CustomGRUModel with the specified number of units and output dimensions.

        Parameters:
        ----------------
        units: int
            Number of units in the CustomGRU block
        output_dim: int
            Number of output units (e.g. 1 for regression tasks)
        
        """
        super(CustomGRUModel,self).__init__()
        self.custom_gru = CustomGRU(units)
        self.dense = Dense(output_dim)

    def call(self,inputs):
        """
        Performs the forward pass of the model 
        Process the input sequence through the CustomGRU block and then applies a dense layer
        to the final hidden state.

        Parameters:
        -----------------
        inputs: Tensor
            Input tensor of shape (batch_size,sequence_length,input_dim)
        
        Returns: 
        -----------------
        Tensor
            The models output predictions

        """

        # Initalize the hidden state to zeros
        batch_size = tf.shape(inputs)[0]
        h = tf.zeros((batch_size,self.custom_gru.units))

        # Process the sequence
        outputs = []
        for t in range(inputs.shape[1]):
            h,_ = self.custom_gru(inputs[:,t,:],[h])
            outputs.append(h)
        
        # Stack outputs and apply dense layer
        outputs = tf.stack(outputs,axis = 1)
        return self.dense(outputs[:,-1,:])



## Example usage

#units = 32 # number of GRU inits
#output_dim = 1  # Single output

# Create an instance of the model
#model = CustomGRUModel(units = units, output_dim=output_dim)

# Compile the model
#model.compile(optimizer = 'adam',loss = 'mse')

# Print model summary
#model.build(input_shape=(None,10,8)) # (batch_size,sequence_length,feature_dim)
#model.summary()

# Genreate dummy data
#import numpy as np
#batch_size = 1000
#sequence_length = 10
#feature_dim= 8

#X_train= np.random.random((batch_size,sequence_length,feature_dim))
#y_train= np.random.random((batch_size,1))

#X_val= np.random.random((200,sequence_length,feature_dim))
#y_val = np.random.random((200,1))

# Train
#model.fit(X_train,y_train, validation_data = (X_val,y_val),epochs = 10,batch_size = 32)