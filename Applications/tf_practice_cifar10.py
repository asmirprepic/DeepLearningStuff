import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """
        Load and prepare the CIFAR-10 dataset.
        Normalizes the images and prepares the data pipeline.
    """
    # Load the CIFAR-10 dataset
    (ds_train,ds_test),ds_info = tfds.load(
        'cifar10',
        split = ['train','test'],
        shuffle_files = True,
        as_supervised = True,
        with_info = True
    )

    # Normalize image to [0,1] range
    def normalize_img(image,label):
        return tf.cast(image,tf.float32)/255.0,label
    
    # Prepare the training dataset
    ds_train = ds_train.map(normalize_img,num_parallel_calls = tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Prepare the test dataset
    ds_test = ds_test.map(normalize_img,num_parallel_calls = tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train,ds_test

def build_model():
    """
        Build a convolutional neural network model.
        Returns:
        A compiled CNN model.
    """
    model = models.Sequential([
        layers.Conv2D(32,(3,3),activation = 'relu',input_shape = (32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3),activation= 'relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3),activation = 'relu'),
        layers.flatten(),
        layers.Dense(64,activation = 'relu'),
        layers.Dense(10)
    ])    

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss = tf.keras.losses.SparseCategoricalCrossEntropy(from_logits = True),
        metrics = ['accuracy']
    )

    return model

def train_model(model,ds_train,ds_test,epochs = 10):
    """
        Train the CNN model.
        Args:
        model: The CNN model to train.
        ds_train: The training dataset.
        ds_test: The test dataset for validation.
        epochs: Number of epochs to train the model.
        Returns:
        Training history.
    """
    history = model.fit(
        ds_train,
        epochs = epochs,
        validation_data = ds_test,
    )

    return history

def evaluate_model(model,ds_test):
    """
        Evaluate the trained model on the test dataset.
        Args:
            model: The trained model.
            ds_test: The test dataset.
        Returns:
            Test loss and accuracy.
    """
    test_loss,test_acc = model.evaluate(ds_test,verbose = 2)
    print(f"Test loss: {test_loss}")
    print(f"\nTest accuracy: {test_acc}")

    
def plot_results(history):
    """
        Plot the training and validation accuracy and loss.
        Args:
        history: Training history.
    """
    plt.figure(figsize=(12,4))

    # Plot training & validation accuracy values
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'],label = 'accuracy')
    plt.plot(history.history['val_accuracy'],label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc = 'low right')
    plt.title('Training and validation accuracy')

    # Plot training and validation loss values
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'],label = 'loss')
    plt.plot(history.history['val_loss'],label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc = 'upper right')
    plt.title('Training and Validation Loss')

    plt.show()

def save_model(model,filename = 'cifar10_cnn_model.h5'):
    """
        Load a saved model from a file.
        Args:
        filename: The filename from which to load the model.
        Returns:
        The loaded model.
    """
    return model.save(filename)

def load_model(filename = 'cifar10_cnn_model.h5'):
    """
        Load a saved model from a file.
        Args:
            filename: The filename from which to load the model.
        Returns:
            The loaded model.
    """
    return tf.keras.models.load_model(filename)

def make_prediction(model,ds_test):
    """
    Make predictions on a batch of test images and visualize the results.
    Args:
        model: The trained model.
        ds_test: The test dataset.
    """
    # Get a batch of test images and labels
    image_batch, label_batch = next(iter(ds_test))

    # Make predictions
    predictions = model.predict(image_batch)

    # Visualize the first 5 test images,true labels and predicted labels
    plt.figure(figsize=(10,10))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(1,5,i+1)
        plt.title(f"True: {label_batch[i]},Pred: {tf.argmax(predictions[i])}")
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    ds_train,ds_test = load_and_prepare_data()
    model = build_model()
    history = train_model(model,ds_train,ds_test,epochs=10)
    evaluate_model(model,ds_test)
    plot_results(history)
    save_model(model)

    make_prediction(model,ds_test)
    

