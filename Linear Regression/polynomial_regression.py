"""
Illustrating a poloynomial regression with tensorflow

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generating synthetic data
np.random.seed(0)
X = np.random.normal(0, 1, 100)
Y = X**2+X**3 + np.random.normal(0, 0.1, 100)  # Cubic relationship

# Preparing data for Polynomial Regression
X_poly = np.column_stack([X, X**2,X**3])

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(3,))
])

model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(X_poly, Y, epochs=100, verbose=0)

# Predictions for visualization
X_test = np.linspace(-3, 3, 100)
X_test_poly = np.column_stack([X_test, X_test**2,X_test**3])
Y_pred = model.predict(X_test_poly)

plt.scatter(X, Y, label='Data')
plt.plot(X_test, Y_pred, color='red', label='Polynomial Regression Fit')
plt.legend()
plt.show()
