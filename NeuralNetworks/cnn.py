"""

Illustration of convolutional neural network

"""

import tensorflow as ts
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as  plt

# Get the data
cifar10 = keras.dataset.cifar10

(train_images,train_labels),(test_images,test_labels) = cifar10.load_data()

# Normalize
train_images,test_images = train_images/255.0, test_images/255.0

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Function to show the image
def show():
  plt.figure(figsize = (10,10))
  for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0])
  plt.show()

# Show images
show()

# Specify model
model = keras.models.Sequential()
model.add(layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
print(model.summary())


# Define loss, optimization and metrics

loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optim  = keras.optimizers.Adam(lr =0.001)
metrics  =['accuracy']

model.compile(optimizer = optim, loss =loss, metrics = metrics)

# traning
batch_size = 64
epochs =5

# Fit model
model.fit(train_images,train_labels,epochs = epochs,batch_size = batch_size,verbose = 2)

# evaluate
model.evaluate(test_images,test_labels,batch_size = batch_size, verbose =2 )














