"""
Illustraties how to create a nerual netowork in tensorflow

"""


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as plt

mnist = keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

## Normalize the data
x_train,x_test = x_train/255.0,x_test/255.0

# Create model
model = keras.meodel.Sequential([
  keras.layers.Flatten(input_shape= (28,28)),
  keras.layers.Dense(128,activation='relu'),
  keras.layers.Dense(10),
])

# Prints the summary of the model
print(model.summary())

# Define loss, optimizer and metrics
loss = keras.losses.SparseCategoricalCrossEntropy(from_logits = True)
optim = keras.optimizers.Adam(lr=0.001)
metrics  = ['accuracy']

# Compile the model
model.compile(loss = loss, optimizer=optim, metrics = metrics)


# Training stuff
batch_size = 64
epochs = 5

# Fit the model
model.fit(x_train,y_train,batch_size = batch_size, epochs = epochs, shuffle = True, verbose =2)

# Evaluate the model
model.evaluate(x_test,y_test, batch_size = batch_size, verbose =2)

# Generate predictions
## This needs to add a output layer to the model to generate the predictions
probability_model = keras.models.Sequential([
  model,
  keras.layers.Softmax()
])
predictions = probability_model(x_text)
pred0 = predictions[0]

# Get label of the prediction with the highest probability
label0 = np.argmax(pred0)
print(label0)










