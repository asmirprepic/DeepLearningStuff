"""
Illustration of a Poisson regression with tensorflow. 
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generating synthetic count data
np.random.seed(0)
n_samples = 100
X = np.random.normal(size=(n_samples, 2))
# Poisson rate parameter (lambda) is exp(linear combination of X)
rate = np.exp(0.5 * X[:, 0] - 0.4 * X[:, 1])
Y = np.random.poisson(rate)

# Poisson Regression Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation=tf.exp, input_shape=(2,))  # Output layer with exponential activation
])

# Custom Poisson loss function
def poisson_loss(y_true, y_pred):
    y_true = tf.cast(y_true,tf.float32)
    return tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred + 1e-7))

model.compile(optimizer='adam', loss=poisson_loss)
model.fit(X, Y, epochs=200, verbose=0)

# Predictions
Y_pred = model.predict(X).flatten()

feature_to_plot = X[:, 0]

# Sort the feature values and corresponding predictions for a nicer plot
sorted_indices = np.argsort(feature_to_plot)
sorted_feature = feature_to_plot[sorted_indices]
sorted_predictions = Y_pred[sorted_indices]

plt.scatter(feature_to_plot, Y, alpha=0.5, label='Actual counts')
plt.scatter(sorted_feature, sorted_predictions, color='red', alpha=0.5, label='Predicted counts')
plt.xlabel('Feature value')
plt.ylabel('Counts')
plt.title('Poisson Regression - Predictions vs. Feature')
plt.legend()
plt.show()
