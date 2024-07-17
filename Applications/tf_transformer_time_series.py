import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

def load_and_preprocess_data(root_url):
    x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
    x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    
    return x_train, y_train, x_test, y_test

def generate_synthetic_time_series_data(num_samples=1000, sequence_length=100, num_classes=2):
    x = np.zeros((num_samples, sequence_length, 1))
    y = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        freq = np.random.rand() * 10
        phase = np.random.rand() * 2 * np.pi
        noise = np.random.normal(0, 0.1, sequence_length)
        x[i, :, 0] = np.sin(np.linspace(0, 2 * np.pi, sequence_length) * freq + phase) + noise
        y[i] = np.random.randint(0, num_classes)
    
    return x, y

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0, n_classes=2):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

def train_and_evaluate_model(x_train, y_train, x_test, y_test, input_shape, n_classes, epochs=150, batch_size=64):
    model = build_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25, n_classes=n_classes)
    
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"]
    )
    model.summary()

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_acc}")
    
    return model, history

def plot_model_performance(history):
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['sparse_categorical_accuracy'], label='accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and validation accuracy')
    
    # Plot training and validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and validation loss')
    
    plt.show()

def save_model(model, filename='transformer_model'):
    model.save(filename)

def load_model(filename='transformer_model'):
    return tf.keras.models.load_model(filename)

if __name__ == '__main__':
    # Use either the synthetic data or real data
    use_synthetic_data = True
    
    if use_synthetic_data:
        x_train, y_train = generate_synthetic_time_series_data(num_samples=1000, sequence_length=100, num_classes=2)
        x_test, y_test = generate_synthetic_time_series_data(num_samples=200, sequence_length=100, num_classes=2)
    else:
        root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
        x_train, y_train, x_test, y_test = load_and_preprocess_data(root_url)
    
    input_shape = x_train.shape[1:]
    n_classes = len(np.unique(y_train))

    model, history = train_and_evaluate_model(x_train, y_train, x_test, y_test, input_shape, n_classes)
    plot_model_performance(history)
    save_model(model)

    loaded_model = load_model('transformer_model')
    print(f"Loaded model validation accuracy: {loaded_model.evaluate(x_test, y_test, verbose=2)[1]}")
