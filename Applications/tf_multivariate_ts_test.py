import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generate synthetic multivariate time series data
def generate_multivariate_data(num_samples=10000, num_features=3):
    np.random.seed(42)
    timestamps = pd.date_range("2023-01-01", periods=num_samples, freq="T")
    data = np.zeros((num_samples, num_features))

    # Generate trend and seasonality for each feature
    for i in range(num_features):
        trend = np.linspace(0, 2, num_samples)
        daily_seasonality = 2 * np.sin(2 * np.pi * (timestamps.hour * 60 + timestamps.minute) / 1440)
        weekly_seasonality = 1 * np.sin(2 * np.pi * timestamps.dayofweek / 7)
        noise = np.random.normal(0, 0.1, num_samples)
        data[:, i] = trend + daily_seasonality + weekly_seasonality + noise

    return pd.DataFrame(data, columns=[f"feature_{i}" for i in range(num_features)], index=timestamps)

def preprocess_and_create_sequences(df, seq_length=60):
    # Normalize features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    data_scaled_df = pd.DataFrame(data_scaled, index=df.index, columns=df.columns)

    # Create sequences
    xs, ys = [], []
    for i in range(len(data_scaled_df) - seq_length):
        x = data_scaled_df.iloc[i:(i + seq_length)].values
        y = data_scaled_df.iloc[i + seq_length].values
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys), scaler

# Generate and preprocess data
data = generate_multivariate_data()
SEQ_LENGTH = 60
X, y, scaler = preprocess_and_create_sequences(data, SEQ_LENGTH)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to TensorFlow datasets
ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(32).prefetch(tf.data.AUTOTUNE)
ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)

def build_multivariate_model(input_shape):
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.LSTM(32),
        layers.Dense(input_shape[-1])
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

def make_multivariate_prediction(model, scaler, data, seq_length):
    last_sequence = data.iloc[-seq_length:].values.reshape((1, seq_length, data.shape[1]))
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, data.shape[1])).reshape(1, seq_length, data.shape[1])

    predicted_values = model.predict(last_sequence_scaled)
    predicted_values_original_scale = scaler.inverse_transform(predicted_values[0])

    print(f"Predicted next values: {predicted_values_original_scale}")

    for i in range(data.shape[1]):
        plt.figure(figsize=(10, 6))
        plt.plot(data[f'feature_{i}'].values, label='Actual Data')
        plt.plot(range(len(data), len(data) + 1), [predicted_values_original_scale[i]], 'ro', label='Predicted Value')
        plt.xlabel('Time')
        plt.ylabel(f'Feature_{i} Value')
        plt.legend()
        plt.title(f'Feature_{i} Time Series and Prediction')
        plt.show()

if __name__ == '__main__':
    input_shape = (SEQ_LENGTH, X_train.shape[2])
    model = build_multivariate_model(input_shape)
    history = train_model(model, ds_train, ds_test, epochs=20)
    evaluate_model(model, ds_test)
    plot_results(history)
    make_multivariate_prediction(model, scaler, data, SEQ_LENGTH)
