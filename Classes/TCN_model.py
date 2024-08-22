import tensorflow as tf
from tensorflow.keras import layers,model

class TCNModel(tf.keras.Model):
    """
    Temporal Convolutional Network (TCN) Model

    This model is designed for sequence modeling taks, such as time series forecasting 
    sequene classification, and anomaly detection. It uses dilated causal convolutions
    to capture temporal dependencies across various time scales. 

    Attributes: 
    --------------
    num_channels: int
        The number of filters in each convolutional layer
    kernel_size: int
        The size of the convolutional kernel
    num_outputs: int
        The number of output units (e.g., 1 for regression tasks)
    """

    def __init__(self,num_channels,kernel_size,num_outputs):
        """
        Initialize the TCN model. 

        Parameters: 
        ---------------
        num_channels: int
            Number of filters in each Conv1D layer
        kernel_size: int
            Size of the convolutional kernel (number of time steps)
        num_outputs: int
            Number of output units (e.g., 1 for regression tasks)
        
        """
        super(TCNModel,self).__init__()
        
        # First dilated convolutional layer with dilation rate = 1
        # This layer captures local temporal patterns

        self.conv1 = layers.Conv1D(
            num_channels,kernel_size,
            padding = 'causal', dilation_rate = 1,
            activation = 'relu'
        )

        # Second dilated convolutional layer with dilation rate = 2
        # This layer captures broader temporal patterns by increasing receptive field. 
        
        self.conv2 = layers.Conv1D(
            num_channels,kernel_size,
            padding = 'causal',dilation_rate = 2,
            activation = 'relu'
        )

        # Third dilated convolutional layer with dilation rate = 4
        # This layer expands the receptive field , capturing long-term dependencies
        self.conv3 = layers.Conv1D(
            num_channels,kernel_size,
            padding = 'causal',dilation_rate = 4,
            activation = 'relu'

        )

        # Global average pooling layer to reduce the dimensionality of the output
        # by averaging across the time dimension.
        self.global_pool = layers.GlobalAveragePooling1D()

        # Dense layer with 128 units and ReLu activation
        # This layer processes the pooled features for final prediction
        self.dense = layers.Dense(128,activation = 'relu')

    def call(self,inputs,training = False):
        """
        Defines the forward pass of the model. 

        Parameters: 
        ---------------
        inputs: Tensor
            Input tensor of shape (batch_size,time_steps,features)
        training: bool,optional
            If the model is in training mode (default is free)
        
        Returns: 
        -------------
        Tensor
            output tensor of shape (batch_size, num_outputs)
    
        """ 

        # Pass the input through the first convolutional layer
        x = self.conv1(inputs)

        # Pass the output through the second convoluational layer
        x= self.conv2(x)

        # Pass the output through the third convolutional layer
        x = self.conv3(x)

        # Apply global average pooling to the output
        x = self.global_pool(x)

        # Pass the pooled output through the dense layer
        x = self.dense(x)

        # Final output
        return self.output_layer(x)
    
# Example usage

num_channels = 64
kernel_size = 3
num_ouputs = 1 # For regression

# Initalize and compile the model
model = TCNModel(num_channels=num_channels,kernel_size=kernel_size,num_outputs=num_ouputs)
model.compile(optimizer = 'adam',loss = 'mse',metrics = ['mae'])

model.build(input_shape = (None,100,64))

model.summary()