"""
Illustrating LSTM for movie reviews 

"""


import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.datasets import imdb

# Parameters
max_features = 10000
maxlen = 500
batch_size = 32

# Load IMDB dataset
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)

# Preprocess data
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

# Build the model
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(input_train, y_train, epochs=10, batch_size=batch_size, validation_split=0.2)

# Evaluate the model
model.evaluate(input_test, y_test)




# Predict on the test set
y_pred = model.predict(input_test).flatten()

# Create a heatmap data
heatmap_data = np.reshape(y_pred, (len(y_pred) // 10, 10))

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap='viridis')
plt.title("Heatmap of Sentiment Predictions")
plt.ylabel("Batch of Reviews")
plt.xlabel("Review Index in Batch")
plt.show()
