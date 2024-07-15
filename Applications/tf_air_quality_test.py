import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data():
    """
    Load and prepare the Air Quality dataset.
    Normalize the data and prepare the data pipeline.
    """
    # Load the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
    data = pd.read_csv(url, sep=';', decimal=',', header=0, na_values=-200, parse_dates={'datetime': [0, 1]}, infer_datetime_format=True, compression='zip')

    # Drop unnecessary columns and rows with missing values
    data = data.drop(['Date', 'Time', 'NMHC(GT)'], axis=1).dropna()

    # Normalize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Convert to DataFrame
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

    # Prepare the sequences
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data.iloc[i:(i + seq_length)].values
            y = data.iloc[i + seq_length].values[0]  # Predicting the CO(GT) value
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    SEQ_LENGTH = 24  # 24 hours
    X, y = create_sequences(data_scaled, SEQ_LENGTH)

    # Split the data into training and test sets
    SPLIT_RATIO = 0.8
    split_index = int(len(X) * SPLIT_RATIO)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Convert to TensorFlow datasets
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(32).prefetch(tf.data.AUTOTUNE)
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test, scaler, data, SEQ_LENGTH

def build_model(input_shape):
    """
    Build an LSTM model for multivariate time series forecasting.
    Returns:
    A compiled model.
    """
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.LSTM(32),
        layers.Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
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
    Test loss.
    """
    test_loss = model.evaluate(ds_test, verbose=2)
    print(f"Test loss: {test_loss}")

def plot_results(history):
    """
    Plot the training and validation loss.
    Args:
    history: Training history.
    """
    plt.figure(figsize=(12, 4))

    # Plot training & validation loss values
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

def make_prediction(model, scaler, data, seq_length):
    """
    Make predictions on the last part of the data and visualize the results.
    Args:
    model: The trained model.
    scaler: The scaler used to normalize the data.
    data: The original data.
    seq_length: Length of the input sequences.
    """
    # Get the last sequence from the data
    last_sequence = data.iloc[-seq_length:].values.reshape((1, seq_length, data.shape[1]))
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, data.shape[1])).reshape(1, seq_length, data.shape[1])

    # Make predictions
    predicted_value = model.predict(last_sequence_scaled)
    predicted_value_original_scale = scaler.inverse_transform([[predicted_value[0][0]] + [0]*(data.shape[1]-1)])[0][0]

    print(f"Predicted next CO(GT) value: {predicted_value_original_scale}")

    # Plot the actual data and the predicted value
    plt.figure(figsize=(10, 6))
    plt.plot(data['CO(GT)'].values, label='Actual Data')
    plt.plot(range(len(data), len(data) + 1), [predicted_value_original_scale], 'ro', label='Predicted Value')
    plt.xlabel('Time')
    plt.ylabel('CO(GT) Concentration')
    plt.legend()
    plt.title('CO(GT) Concentration Time Series and Prediction')
    plt.show()

if __name__ == '__main__':
    ds_train, ds_test, scaler, data, seq_length = load_and_prepare_data()
    input_shape = (seq_length, data.shape[1])
    model = build_model(input_shape)
    history = train_model(model, ds_train, ds_test, epochs=20)
    evaluate_model(model, ds_test)
    plot_results(history)

    make_prediction(model, scaler, data, seq_length)
