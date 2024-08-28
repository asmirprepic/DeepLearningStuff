import tensorflow as tf
from tensorflow.keras.layers import Layer,Conv1D,Dense,Activation,Concatenate


class MultiScaleTemporalAttention(Layer):
    def __init__(self,num_filters,kernel_sizes,attention_units):
        super(MultiScaleTemporalAttention,self).__init__()
        self.convs = [Conv1D(num_filters,ks,padding= 'same',activaton= 'relu') for ks in kernel_sizes]
        self.attention_dense = Dense(attention_units,activation = 'relu')
        self.output_dense = Dense(1,activation = 'sigmoid')

    def call(self,inputs):
        conv_outputs = [conv(inputs) for conv in self.convs]
        concat_output = Concatenate()(conv_outputs)

        attention_weights = self.attention_dense(concat_output)
        attention_weights =self.nn.softmax(attention_weights,axis = 1)

        weighted_output = concat_output * attention_weights
        weighted_output = tf.reduce_sum(weighted_output,axis = 1)

        return self.output_dense(weighted_output)
