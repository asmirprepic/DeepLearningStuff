import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Simulate high-frequency time series data with trends and seasonality
def generate_data(num_samples=10000, num_features=5):
    np.random.seed(42)
    timestamps = pd.date_range("2023-01-01", periods=num_samples, freq="T")

    # Generate trends
    trend = np.linspace(0, 10, num_samples)

    # Generate seasonality (daily and weekly patterns)
    daily_seasonality = 10 * np.sin(2 * np.pi * (timestamps.hour * 60 + timestamps.minute) / 1440)
    weekly_seasonality = 5 * np.sin(2 * np.pi * timestamps.dayofweek / 7)

    # Combine trend, seasonality, and noise
    numerical_data = trend.reshape(-1, 1) + daily_seasonality.reshape(-1, 1) + weekly_seasonality.reshape(-1, 1) + np.random.randn(num_samples, num_features)
    
    categorical_data = np.random.choice(["A", "B", "C"], size=num_samples)
    return pd.DataFrame(
        np.hstack([numerical_data, categorical_data.reshape(-1, 1)]),
        columns=[f"num_{i}" for i in range(num_features)] + ["cat"],
        index=timestamps
    )

def preprocess_and_create_sequences(df, seq_length=60):
    # One-hot encode categorical features
    cat_encoder = OneHotEncoder(sparse=False)
    cat_encoded = cat_encoder.fit_transform(df[['cat']])
    cat_encoded_df = pd.DataFrame(cat_encoded, index=df.index, columns=cat_encoder.get_feature_names_out())

    # Normalize numerical features
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(df.drop('cat', axis=1))
    num_scaled_df = pd.DataFrame(num_scaled, index=df.index, columns=df.columns[:-1])

    # Combine numerical and categorical features
    data_scaled = pd.concat([num_scaled_df, cat_encoded_df], axis=1)

    # Create sequences
    xs, ys = [], []
    for i in range(len(data_scaled) - seq_length):
        x = data_scaled.iloc[i:(i + seq_length)].values
        y = data_scaled.iloc[i + seq_length].values[0]  # Predicting the first numerical feature
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys), scaler, cat_encoder

# Generate and preprocess data
data = generate_data()
SEQ_LENGTH = 60  # 1 hour of data
X, y, scaler, cat_encoder = preprocess_and_create_sequences(data, SEQ_LENGTH)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to TensorFlow datasets
ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(32).prefetch(tf.data.AUTOTUNE)
ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)

def build_model(input_shape):
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

def make_prediction(model, scaler, data, seq_length):
    last_sequence = data.iloc[-seq_length:].values.reshape((1, seq_length, data.shape[1]))
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, data.shape[1])).reshape(1, seq_length, data.shape[1])

    predicted_value = model.predict(last_sequence_scaled)
    predicted_value_original_scale = scaler.inverse_transform([[predicted_value[0][0]] + [0]*(data.shape[1]-1)])[0][0]

    print(f"Predicted next num_0 value: {predicted_value_original_scale}")

    plt.figure(figsize=(10, 6))
    plt.plot(data['num_0'].values, label='Actual Data')
    plt.plot(range(len(data), len(data) + 1), [predicted_value_original_scale], 'ro', label='Predicted Value')
    plt.xlabel('Time')
    plt.ylabel('num_0 Value')
    plt.legend()
    plt.title('num_0 Time Series and Prediction')
    plt.show()

if __name__ == '__main__':
    input_shape = (SEQ_LENGTH, X_train.shape[2])
    model = build_model(input_shape)
    history = train_model(model, ds_train, ds_test, epochs=20)
    evaluate_model(model, ds_test)
    plot_results(history)
    make_prediction(model, scaler, data, SEQ_LENGTH)
