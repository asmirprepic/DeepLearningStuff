"""
Illustrating qunatile regression with tensorflow

"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Features
Y = 2.5 * X + np.random.normal(0, 2, size=(100, 1))  # Target with noise

# Define the quantile loss function
def quantile_loss(q, y_true, y_pred):
    e = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# Quantile to predict
quantile = 0.9

# Compile the model with the custom loss function
model.compile(optimizer='adam', loss=lambda y_true, y_pred: quantile_loss(quantile, y_true, y_pred))

# Train the model
model.fit(X, Y, epochs=1000, verbose=0)

# Predictions
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
Y_pred = model.predict(X_test)

# Visualization
plt.scatter(X, Y, alpha=0.5, label='Data')
plt.plot(X_test, Y_pred, color='red', label=f'Quantile Regression (q={quantile})')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Quantile Regression')
plt.legend()
plt.show()
