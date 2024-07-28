import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

class MyTrainer(Model):
    def __init__(self, model, loss_fn=None):
        super(MyTrainer, self).__init__()
        self.model = model
        # Allow custom loss function or use default
        self.loss_fn = loss_fn if loss_fn else SparseCategoricalCrossentropy()
        self.accuracy_metric = SparseCategoricalAccuracy()

    @property
    def metrics(self):
        # List metrics here.
        return [self.accuracy_metric]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)  # Forward pass
            # Compute loss value
            loss = self.loss_fn(y, y_pred)

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.accuracy_metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value.
        return {"loss": loss, self.accuracy_metric.name: self.accuracy_metric.result()}

    def test_step(self, data):
        x, y = data

        # Inference step
        y_pred = self.model(x, training=False)

        # Update metrics
        self.accuracy_metric.update_state(y, y_pred)
        return {self.accuracy_metric.name: self.accuracy_metric.result()}

    def call(self, x):
        # Equivalent to `call()` of the wrapped keras.Model
        return self.model(x)

# Example usage with a custom loss function
def custom_loss_function(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))  # Mean Squared Error for demonstration

input_shape = (28, 28, 1)
num_classes = 10

# Create a simple model for demonstration
inputs = tf.keras.Input(shape=input_shape)
x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
x = tf.keras.layers.Flatten()(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

trainer = MyTrainer(model, loss_fn=custom_loss_function)
trainer.compile(optimizer=tf.keras.optimizers.Adam())

# Dummy dataset for demonstration
import numpy as np

x_train = np.random.random((100, 28, 28, 1))
y_train = np.random.randint(0, num_classes, 100)
x_val = np.random.random((20, 28, 28, 1))
y_val = np.random.randint(0, num_classes, 20)

trainer.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5)
