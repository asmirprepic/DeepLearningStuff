import os
import tensorflow as tf
import pandas as pd
import keras
from keras import layers, models
import matplotlib.pyplot as plt

# Ensure TensorFlow backend is used
os.environ["KERAS_BACKEND"] = "tensorflow"

def load_data(file_url):
    """
    Load and prepare the heart disease dataset.
    """
    dataframe = pd.read_csv(file_url)
    val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
    train_dataframe = dataframe.drop(val_dataframe.index)
    print(f"Using {len(train_dataframe)} samples for training and {len(val_dataframe)} for validation")
    return train_dataframe, val_dataframe

def dataframe_to_dataset(dataframe):
    """
    Convert dataframe to TensorFlow dataset.
    """
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

def preprocess_datasets(train_dataframe, val_dataframe):
    """
    Prepare training and validation datasets.
    """
    train_ds = dataframe_to_dataset(train_dataframe).batch(32).cache().prefetch(tf.data.AUTOTUNE)
    val_ds = dataframe_to_dataset(val_dataframe).batch(32).cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

def encode_numerical_feature(feature, name, dataset):
    normalizer = layers.Normalization()
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))
    normalizer.adapt(feature_ds)
    encoded_feature = normalizer(feature)
    return encoded_feature

def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = layers.StringLookup if is_string else layers.IntegerLookup
    lookup = lookup_class(output_mode="binary")
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))
    lookup.adapt(feature_ds)
    encoded_feature = lookup(feature)
    return encoded_feature

def create_inputs():
    """
    Create input layers for the model.
    """
    return {
        "sex": keras.Input(shape=(1,), name="sex", dtype="int64"),
        "cp": keras.Input(shape=(1,), name="cp", dtype="int64"),
        "fbs": keras.Input(shape=(1,), name="fbs", dtype="int64"),
        "restecg": keras.Input(shape=(1,), name="restecg", dtype="int64"),
        "exang": keras.Input(shape=(1,), name="exang", dtype="int64"),
        "ca": keras.Input(shape=(1,), name="ca", dtype="int64"),
        "thal": keras.Input(shape=(1,), name="thal", dtype="string"),
        "age": keras.Input(shape=(1,), name="age"),
        "trestbps": keras.Input(shape=(1,), name="trestbps"),
        "chol": keras.Input(shape=(1,), name="chol"),
        "thalach": keras.Input(shape=(1,), name="thalach"),
        "oldpeak": keras.Input(shape=(1,), name="oldpeak"),
        "slope": keras.Input(shape=(1,), name="slope")
    }

def encode_features(inputs, train_ds):
    """
    Encode all features.
    """
    encoded_features = []
    for name, input in inputs.items():
        if name in ["sex", "cp", "fbs", "restecg", "exang", "ca"]:
            encoded_features.append(encode_categorical_feature(input, name, train_ds, False))
        elif name == "thal":
            encoded_features.append(encode_categorical_feature(input, name, train_ds, True))
        else:
            encoded_features.append(encode_numerical_feature(input, name, train_ds))
    return layers.concatenate(encoded_features)

def build_model(train_ds):
    """
    Build and compile the model.
    """
    inputs = create_inputs()
    encoded_features = encode_features(inputs, train_ds)
    x = layers.Dense(32, activation="relu")(encoded_features)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_model(model, ds_train, ds_val, epochs=10):
    """
    Train the model.
    """
    history = model.fit(ds_train, validation_data=ds_val, epochs=epochs)
    return history

def evaluate_model(model, ds_test):
    """
    Evaluate the model on the test dataset.
    """
    test_loss, test_acc = model.evaluate(ds_test, verbose=2)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_acc}")

def plot_results(history):
    """
    Plot training and validation accuracy and loss.
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

def save_model(model, filename='heart_disease_model.h5'):
    """
    Save the trained model to a file.
    """
    model.save(filename)

def load_model(filename='heart_disease_model.h5'):
    """
    Load a saved model from a file.
    """
    return tf.keras.models.load_model(filename)

if __name__ == '__main__':
    file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
    train_dataframe, val_dataframe = load_data(file_url)
    ds_train, ds_val = preprocess_datasets(train_dataframe, val_dataframe)
    model = build_model(ds_train)
    history = train_model(model, ds_train, ds_val, epochs=10)
    evaluate_model(model, ds_val)
    plot_results(history)
    save_model(model)

