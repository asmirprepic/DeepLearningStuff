import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Simulate high-frequency Poisson count time series data with trends and seasonality
def generate_poisson_data(num_samples=10000, num_features=5, base_rate=5):
    np.random.seed(42)
    timestamps = pd.date_range("2023-01-01", periods=num_samples, freq="T")

    # Generate trends
    trend = np.linspace(0, 2, num_samples)

    # Generate seasonality (daily and weekly patterns)
    daily_seasonality = 2 * np.sin(2 * np.pi * (timestamps.hour * 60 + timestamps.minute) / 1440)
    weekly_seasonality = 1 * np.sin(2 * np.pi * timestamps.dayofweek / 7)

    # Combine trend, seasonality, and noise
    rate = base_rate + trend + daily_seasonality + weekly_seasonality
    rate = np.maximum(rate, 0)  # Ensure rate is non-negative

    # Generate Poisson-distributed data
    poisson_data = np.random.poisson(rate, size=(num_samples, num_features))
    
    categorical_data = np.random.choice(["A", "B", "C"], size=num_samples)
    return pd.DataFrame(
        np.hstack([poisson_data, categorical_data.reshape(-1, 1)]),
        columns=[f"count_{i}" for i in range(num_features)] + ["cat"],
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
data = generate_poisson_data()
SEQ_LENGTH = 60  # 1 hour of data
X, y, scaler, cat_encoder = preprocess_and_create_sequences(data, SEQ_LENGTH)

# Split the data into training and test sets
X_train, X_test, y_train,
