
"""

Illustrating how to create a General Adveserial Model to create new images form the mnist data set
"""


import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess data
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_train = (x_train - 127.5) / 127.5  # Normalize the images to [-1, 1]

# Buffer size and batch size
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Create the models
def make_generator_model():
    model = tf.keras.Sequential()
    
    # First Dense layer: takes a random noise vector as input and expands it.
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))

    # BatchNormalization: stabilizes training by normalizing the input to each activation function.
    model.add(layers.BatchNormalization())

    # LeakyReLU: avoids the dying ReLU problem, where neurons can stop learning.
    model.add(layers.LeakyReLU())

    # Reshape layer: reshapes the output into a 3D format for convolutional operations.
    model.add(layers.Reshape((7, 7, 256)))

    # Conv2DTranspose: upsamples the input to a larger spatial dimension.
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Additional Conv2DTranspose layers for further upsampling.
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Final Conv2DTranspose layer: upsamples to the size of the target image.
    # 'tanh' activation outputs pixel values in the range [-1, 1].
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()

    # First Conv2D layer: takes an image as input and applies convolution for feature extraction.
    # Strides reduce the spatial dimensions (downsampling).
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())

    # Dropout: prevents overfitting by randomly setting a fraction of the input units to 0.
    model.add(layers.Dropout(0.3))

    # Additional Conv2D layers: further convolution and downsampling.
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Flatten layer: converts the 3D feature maps to 1D feature vectors.
    model.add(layers.Flatten())

    # Final Dense layer: outputs a single value representing the discriminator's belief 
    # in whether the input is real or fake.
    model.add(layers.Dense(1))

    return model


generator = make_generator_model()
discriminator = make_discriminator_model()


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime to visualize progress
seed = tf.random.normal([num_examples_to_generate, noise_dim])

EPOCHS = 50  # Number of epochs to train for
noise_dim = 100  # Dimensionality of the noise vector for the generator
num_examples_to_generate = 16  # Number of examples to generate for visualization

# Seed for generating consistent images to visualize progress
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function  # Decorator to compile the train_step into a TensorFlow graph for faster execution
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])  # Random noise for generator input

    # GradientTape to record differentiation operations for automatic differentiation
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)  # Generate fake images

        real_output = discriminator(images, training=True)  # Discriminator output for real images
        fake_output = discriminator(generated_images, training=True)  # Discriminator output for fake images

        gen_loss = generator_loss(fake_output)  # Calculate generator loss
        disc_loss = discriminator_loss(real_output, fake_output)  # Calculate discriminator loss

    # Calculate gradients for generator and discriminator
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Apply gradients to optimizer, updating the model weights
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):  # Loop over the dataset for a number of epochs
        for image_batch in dataset:  # Loop over each batch in the dataset
            train_step(image_batch)  # Perform one training step

        # Produce and save images after each epoch for visualization
        generate_and_save_images(generator, epoch + 1, seed)

    # Generate and save images after the final epoch
    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)  # Generate images from the model
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):  # Loop over each image in the batch
        plt.subplot(4, 4, i+1)  # Create a subplot for each image
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')  # Display the image
        plt.axis('off')  # Turn off axis labels

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))  # Save the current figure
    plt.show()  # Display the figure


train(train_dataset, EPOCHS)  # Train the GAN for the specified number of epochs




