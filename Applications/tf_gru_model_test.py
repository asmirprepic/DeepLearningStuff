import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class GRUModel:
    def __init__(self, time_step=10, units=50, dropout_rate=0.2):
        self.time_step = time_step
        self.units = units
        self.dropout_rate = dropout_rate
        self.model = None

    def create_dataset(self, data):
        X, Y = [], []
        for i in range(len(data) - self.time_step - 1):
            X.append(data[i:(i + self.time_step), 0])
            Y.append(data[i + self.time_step, 0])
        return np.array(X), np.array(Y)

    def build_model(self):
        model = Sequential()
        model.add(GRU(self.units, return_sequences=True, input_shape=(self.time_step, 1)))
        model.add(GRU(self.units, return_sequences=False))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def fit(self, returns, epochs=1, batch_size=1, test_split=0.3):
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(returns.reshape(-1, 1))
        X, Y = self.create_dataset(returns_scaled)

        X = X.reshape(X.shape[0], X.shape[1], 1)

        train_size = int(len(X) * (1 - test_split))
        X_train, X_test = X[:train_size], X[train_size:]
        Y_train, Y_test = Y[:train_size], Y[train_size:]

        self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

        return X_test, Y_test, scaler

    def predict(self, X, scaler):
        predictions = self.model.predict(X)
        return scaler.inverse_transform(predictions)

# Example usage
gru_model = GRUModel()
gru_model.build_model()
X_test, Y_test, scaler = gru_model.fit(data, epochs=10)
gru_predictions = gru_model.predict(X_test, scaler)

# Plot GRU predicted volatility
plt.plot(scaler.inverse_transform(data.reshape(-1, 1)))
plt.plot(np.arange(gru_model.time_step, len(gru_predictions) + gru_model.time_step), gru_predictions, label='GRU Predict')
plt.title('GRU Predicted')
plt.xlabel('Time')
plt.legend()
plt.show()
