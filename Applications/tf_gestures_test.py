import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """
    Load and prepare the SmartWatch Gestures dataset.
    Normalize the sensor data and prepare the data pipeline.
    """
    # Load the SmartWatch Gestures dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'smartwatch_gestures',  # Replace with the actual dataset name if different
        split=['train', 'test'],
        shuffle_files=True,
        with_info=True
    )

    # Function to extract features and normalize
    def extract_and_normalize(features):
        accel_x = features['features']['accel_x']
        accel_y = features['features']['accel_y']
        accel_z = features['features']['accel_z']
        gesture = features['gesture']
        
        # Stack the accelerometer data
        accel_data = tf.stack([accel_x, accel_y, accel_z], axis=-1)
        
        # Normalize the data
        accel_data = tf.cast(accel_data, tf.float32)
        accel_data = (accel_data - tf.reduce_mean(accel_data)) / tf.math.reduce_std(accel_data)

        return accel_data, gesture

    # Prepare the training dataset
    ds_train = ds_train.map(lambda x: extract_and_normalize(x), num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.padded_batch(128, padded_shapes=([None, 3], []))
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Prepare the test dataset
    ds_test = ds_test.map(lambda x: extract_and_normalize(x), num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.padded_batch(128, padded_shapes=([None, 3], []))
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test

# Load and prepare the dataset
ds_train, ds_test = load_and_prepare_data()




def build_and_train_model(ds_train, ds_test):
    """
    Build and train a gesture recognition model.
    """
    model = models.Sequential([
        layers.Masking(mask_value=0., input_shape=(None, 3)),  # Mask padding values
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(20, activation='softmax')  # 20 different gestures
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(ds_train, epochs=10, validation_data=ds_test)

    return model, history

# Build and train the model
model, history = build_and_train_model(ds_train, ds_test)

def evaluate_model(model, ds_test):
    """
    Evaluate the trained model on the test dataset.
    Args:
        model: The trained model.
        ds_test: The test dataset.
    Returns:
        Test loss and accuracy.
    """
    test_loss, test_acc = model.evaluate(ds_test, verbose=2)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_acc}")

# Evaluate the model
evaluate_model(model, ds_test)

def plot_results(history):
    """
    Plot the training and validation accuracy and loss.
    Args:
        history: Training history.
    """
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

# Plot training results
plot_results(history)

def make_prediction(model, ds_test):
    """
    Make predictions on a batch of test data and visualize the results.
    Args:
        model: The trained model.
        ds_test: The test dataset.
    """
    for features, labels in ds_test.take(1):
        predictions = model.predict(features)
        predicted_labels = tf.argmax(predictions, axis=1)

        # Visualize the results
        plt.figure(figsize=(10, 10))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.plot(features[i].numpy())
            plt.title(f"True: {labels[i]}, Pred: {predicted_labels[i]}")
            plt.axis('off')
        plt.show()

# Example usage
make_prediction(model, ds_test)
