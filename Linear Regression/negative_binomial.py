import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial

def generate_negative_binomial_data(n_samples, n_features, weights, dispersion, seed=0):
    np.random.seed(seed)
    X = np.random.normal(size=(n_samples, n_features))
    linear_combination = np.dot(X, weights)
    rate = np.exp(linear_combination)
    Y = np.random.negative_binomial(dispersion, rate / (rate + dispersion))
    return X, Y

def build_negative_binomial_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, kernel_initializer='he_normal', input_shape=input_shape)
    ])
    return model

def negative_binomial_loss_fn(y_true, y_pred, dispersion):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.maximum(y_pred, 1e-7)  # Ensure y_pred is non-zero to prevent log(0)

    # Calculate the log of (dispersion + predicted rate)
    log_dispersion_plus_mu = tf.math.log(dispersion + y_pred)

    # Log-gamma functions for stability in gamma functions
    log_gamma_dispersion = tf.math.lgamma(dispersion)
    log_gamma_y_true_plus_dispersion = tf.math.lgamma(y_true + dispersion)
    log_gamma_y_true_plus_one = tf.math.lgamma(y_true + 1)

    # Compute the negative binomial log likelihood
    term1 = log_gamma_y_true_plus_dispersion - log_gamma_dispersion - log_gamma_y_true_plus_one
    term2 = dispersion * (tf.math.log(dispersion) - log_dispersion_plus_mu)
    term3 = y_true * (tf.math.log(y_pred) - log_dispersion_plus_mu)

    return tf.reduce_mean(term1 + term2 + term3)

def negative_binomial_loss(dispersion):
    return partial(negative_binomial_loss_fn, dispersion=dispersion)

def plot_results(X, Y, Y_pred, feature_idx=0):
    feature_to_plot = X[:, feature_idx]
    
    # Sort the feature values and corresponding predictions for a nicer plot
    sorted_indices = np.argsort(feature_to_plot)
    sorted_feature = feature_to_plot[sorted_indices]
    sorted_predictions = Y_pred[sorted_indices]
    
    # Plotting
    plt.scatter(feature_to_plot, Y, alpha=0.5, label='Actual counts')
    plt.scatter(sorted_feature, sorted_predictions, color='red', alpha=0.5, label='Predicted counts')
    plt.xlabel(f'Feature {feature_idx}')
    plt.ylabel('Counts')
    plt.title('Negative Binomial Regression - Predictions vs. Feature')
    plt.legend()
    plt.show()

# Parameters
n_samples = 100
n_features = 2
weights = np.array([0.5, -0.4])
dispersion = 2.0

# Generate data
X, Y = generate_negative_binomial_data(n_samples, n_features, weights, dispersion)

# Build and train model
input_shape = (n_features,)
model = build_negative_binomial_model(input_shape)
model.compile(optimizer='adam', loss=negative_binomial_loss(dispersion))
model.fit(X, Y, epochs=200, verbose=0)

# Predictions
Y_pred = model.predict(X).flatten()

# Plot results
plot_results(X, Y, Y_pred)
