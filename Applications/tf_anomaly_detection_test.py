import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generate synthetic time series data with anomalies
def generate_data_with_anomalies(num_samples=10000):
    np.random.seed(42)
    timestamps = pd.date_range("2023-01-01", periods=num_samples, freq="T")
    trend = np.linspace(0, 2, num_samples)
    seasonality = 2 * np.sin(2 * np.pi * (timestamps.hour * 60 + timestamps.minute) / 1440)
    noise = np.random.normal(0, 0.1, num_samples)
    anomalies = np.random.choice([0, 1], size=num_samples, p=[0.99, 0.01])
    data = trend + seasonality + noise + anomalies * 5  # Add large anomalies

    return pd.DataFrame(data, columns=["value"], index=timestamps)

def preprocess_and_create_sequences(df, seq_length=60):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    data_scaled_df = pd.DataFrame(data_scaled, index=df.index, columns=df.columns)

    xs = []
    for i in range(len(data_scaled_df) - seq_length):
        x = data_scaled_df.iloc[i:(i + seq_length)].values
        xs.append(x)

    return np.array(xs), scaler

# Generate and preprocess data
data = generate_data_with_anomalies()
SEQ_LENGTH = 60
X, scaler = preprocess_and_create_sequences(data, SEQ_LENGTH)

# Split the data into training and test sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Convert to TensorFlow datasets
ds_train = tf.data.Dataset.from_tensor_slices(X_train).shuffle(len(X_train)).batch(32).prefetch(tf.data.AUTOTUNE)
ds_test = tf.data.Dataset.from_tensor_slices(X_test).batch(32).prefetch(tf.data.AUTOTUNE)

def build_autoencoder(input_shape):
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.LSTM(32, return_sequences=False),
        layers.RepeatVector(input_shape[0]),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(64, return_sequences=True),
        layers.TimeDistributed(layers.Dense(input_shape[1]))
    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    return model

def train_model(model, ds_train, ds_test, epochs=20):
    history = model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
    )
    return history

def evaluate_model(model, ds_test):
    test_loss = model.evaluate(ds_test, verbose=2)
    print(f"Test loss: {test_loss}")

def plot_results(history):
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def detect_anomalies(model, scaler, data, seq_length, threshold=3.0):
    xs = []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].values
        xs.append(x)

    xs_scaled = scaler.transform(xs)
    xs_scaled = np.array(xs_scaled)

    reconstructed = model.predict(xs_scaled)
    reconstruction_error = np.mean(np.abs(reconstructed - xs_scaled), axis=(1, 2))

    anomalies = reconstruction_error > threshold

    plt.figure(figsize=(10, 6))
    plt.plot(data.values, label='Actual Data')
    plt.plot(np.arange(seq_length, len(data)), anomalies * data.values[seq_length:], 'ro', markersize=3, label='Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Time Series Data with Anomalies')
    plt.show()

if __name__ == '__main__':
    input_shape = (SEQ_LENGTH, X_train.shape[2])
    model = build_autoencoder(input_shape)
    history = train_model(model, ds_train, ds_test, epochs=20)
    evaluate_model(model, ds_test)
    plot_results(history)
    detect_anomalies(model, scaler, data, SEQ_LENGTH)
