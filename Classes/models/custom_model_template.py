import tensorflow as tf
from tensorflow.keras import layers

class CustomModel(tf.keras.Model):
    """
    A template for building a keras model
    """

    def __init__(self,num_classes,hidden_units = 64,dropout_rate = 0.3 ,**kwargs):
        """
        Initialize the custom model
        
        Args:
        ------------
        num_classes (int): Number of output classes (for classification) or output features (for regression)
        hidden_units (int): Number of units in the hidden layers. 
        droupout_rate (float): Dropout rate regularization.
        kwargs: Addiational keyword arguments for the Model Class
        """

        super(CustomModel,self).__init__(**kwargs)

        # Define layers here
        self.dense1 = layers.Dense(hidden_units,activation = 'relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(num_classes)

    def call(self,inputs,training = False):
        """
        Forward pass of the model

        Args:
        ----------
            inputs (Tensor): Input tensor to the model
            tranining (bool): Whtere to run in traning mode or inference mode (useful for dropout)
        
        Returns:
        -----------
            Tensor: the output tensor after applying the models operations.
        """

        x = self.dense1(inputs)
        x = self.dropout(x, training = training)
        return self.dense2(x)
    
    def model(self):
        """
        Builds the full model architecture for vizualization

        Returns:
            Model: A keras model instance

        """

        x= layers.Input(shape = (None,))
        return tf.keras.Model(inputs = [x],outputs =self.call(x))

        