import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial

def generate_gamma_data(n_samples, n_features, weights, shape_param, seed=0):
    np.random.seed(seed)
    X = np.random.normal(size=(n_samples, n_features))
    linear_combination = np.dot(X, weights)
    rate = np.exp(linear_combination)
    Y = np.random.gamma(shape=shape_param, scale=1/rate)
    return X, Y

def build_gamma_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, kernel_initializer='he_normal', input_shape=input_shape)
    ])
    return model

def gamma_loss_fn(y_true, y_pred, shape_param):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.maximum(y_pred, 1e-7)  # Ensure y_pred is non-zero to prevent log(0)

    log_shape = tf.math.log(shape_param)
    log_rate = tf.math.log(y_pred)
    log_y_true = tf.math.log(y_true)

    # Compute the Gamma log likelihood
    term1 = shape_param * log_rate
    term2 = (shape_param - 1) * log_y_true
    term3 = y_true * y_pred
    term4 = tf.math.lgamma(shape_param)

    return tf.reduce_mean(term1 + term2 - term3 - term4)

def gamma_loss(shape_param):
    return partial(gamma_loss_fn, shape_param=shape_param)

def plot_results(X, Y, Y_pred, feature_idx=0):
    feature_to_plot = X[:, feature_idx]
    
    # Sort the feature values and corresponding predictions for a nicer plot
    sorted_indices = np.argsort(feature_to_plot)
    sorted_feature = feature_to_plot[sorted_indices]
    sorted_predictions = Y_pred[sorted_indices]
    
    # Plotting
    plt.scatter(feature_to_plot, Y, alpha=0.5, label='Actual values')
    plt.scatter(sorted_feature, sorted_predictions, color='red', alpha=0.5, label='Predicted values')
    plt.xlabel(f'Feature {feature_idx}')
    plt.ylabel('Values')
    plt.title('Gamma Regression - Predictions vs. Feature')
    plt.legend()
    plt.show()

# Parameters
n_samples = 100
n_features = 2
weights = np.array([0.5, -0.4])
shape_param = 2.0

# Generate data
X, Y = generate_gamma_data(n_samples, n_features, weights, shape_param)

# Build and train model
input_shape = (n_features,)
model = build_gamma_model(input_shape)
model.compile(optimizer='adam', loss=gamma_loss(shape_param))
model.fit(X, Y, epochs=200, verbose=0)

# Predictions
Y_pred = model.predict(X).flatten()

# Plot results
plot_results(X, Y, Y_pred)
