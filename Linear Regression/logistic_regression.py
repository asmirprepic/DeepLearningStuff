"""
Illustrating logistic regression with tensorflow
"""


import tensorflow as tf
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

# Generating synthetic classification data
X, Y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_clusters_per_class=1)

# Logistic Regression Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))
])

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, Y, epochs=100, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X, Y, verbose=0)
print(f"Accuracy: {accuracy:.4f}")

# Function to plot decision boundary
def plot_decision_boundary(X, Y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')

plot_decision_boundary(X, Y, model)
plt.show()
