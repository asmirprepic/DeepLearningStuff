import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """
    Load and prepare the IMDb dataset.
    Tokenizes the text and prepares the data pipeline.
    """
    # Load the IMDb dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'imdb_reviews',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    # Tokenize the text
    tokenizer = tfds.deprecated.text.Tokenizer()

    def encode(text_tensor, label):
        encoded_text = tokenizer.tokenize(text_tensor.numpy())
        return encoded_text, label

    def encode_map_fn(text, label):
        encoded_text, label = tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))
        encoded_text.set_shape([None])
        label.set_shape([])
        return encoded_text, label

    ds_train = ds_train.map(encode_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.padded_batch(32, padded_shapes=([None], []))
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(encode_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.padded_batch(32, padded_shapes=([None], []))
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test, tokenizer

def build_model(vocab_size):
    """
    Build a neural network model for text classification.
    Returns:
    A compiled model.
    """
    model = models.Sequential([
        layers.Embedding(vocab_size, 64),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model

def train_model(model, ds_train, ds_test, epochs=10):
    """
    Train the model.
    Args:
    model: The model to train.
    ds_train: The training dataset.
    ds_test: The test dataset for validation.
    epochs: Number of epochs to train the model.
    Returns:
    Training history.
    """
    history = model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
    )

    return history

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
    print(f"\nTest accuracy: {test_acc}")

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
    plt.title('Training and validation accuracy')

    # Plot training and validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

def save_model(model, filename='imdb_text_classification_model.h5'):
    """
    Save the trained model to a file.
    Args:
    model: The trained model.
    filename: The filename to save the model to.
    """
    model.save(filename)

def load_model(filename='imdb_text_classification_model.h5'):
    """
    Load a saved model from a file.
    Args:
    filename: The filename from which to load the model.
    Returns:
    The loaded model.
    """
    return tf.keras.models.load_model(filename)

def make_prediction(model, tokenizer, raw_texts):
    """
    Make predictions on a batch of raw text reviews and visualize the results.
    Args:
    model: The trained model.
    tokenizer: The tokenizer used to preprocess the text.
    raw_texts: A list of raw text reviews.
    """
    encoded_texts = [tokenizer.tokenize(text) for text in raw_texts]
    padded_texts = tf.keras.preprocessing.sequence.pad_sequences(encoded_texts, padding='post')

    predictions = model.predict(padded_texts)

    for i, text in enumerate(raw_texts):
        print(f"Review: {text}")
        print(f"Predicted Sentiment: {'Positive' if predictions[i] > 0 else 'Negative'}\n")

if __name__ == '__main__':
    ds_train, ds_test, tokenizer = load_and_prepare_data()
    vocab_size = tokenizer.vocab_size
    model = build_model(vocab_size)
    history = train_model(model, ds_train, ds_test, epochs=10)
    evaluate_model(model, ds_test)
    plot_results(history)
    save_model(model)

    raw_texts = [
        "This movie was fantastic! I really enjoyed it.",
        "Terrible film. It was a waste of time.",
        "It was an okay movie, not the best but not the worst."
    ]
    make_prediction(model, tokenizer, raw_texts)
