import numpy as np

class NeuralNetwork:
    def __init__(self, layers=[2, 8, 1], learning_rate=0.01, epochs=100, activation='relu'):
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_type = activation
        self.weights = []
        self.biases = []
        self.history = []
        
        for i in range(len(layers) - 1):
            limit = np.sqrt(2 / layers[i])
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * limit)
            self.biases.append(np.zeros((1, layers[i+1])))

    def _activation(self, x):
        if self.activation_type == 'relu':
            return np.maximum(0, x)
        elif self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        elif self.activation_type == 'tanh':
            return np.tanh(x)
        elif self.activation_type == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        return x

    def _activation_derivative(self, x):
        if self.activation_type == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation_type == 'sigmoid':
            s = self._activation(x)
            return s * (1 - s)
        elif self.activation_type == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation_type == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)
        return 1

    def forward(self, X):
        activations = [X]
        zs = []
        
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            if i == len(self.weights) - 1:
                a = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
            else:
                a = self._activation(z)
            activations.append(a)
            
        return activations, zs

    def backward(self, activations, zs, y):
        m = y.shape[0]
        y = y.reshape(-1, 1)
        
        da = activations[-1] - y
        
        for i in reversed(range(len(self.weights))):
            dw = np.dot(activations[i].T, da) / m
            db = np.sum(da, axis=0, keepdims=True) / m
            
            if i > 0:
                da = np.dot(da, self.weights[i].T) * self._activation_derivative(zs[i-1])
            
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

    def fit(self, X, y, grid_X=None):
        n_snapshots = min(20, self.epochs)
        snapshot_interval = max(1, self.epochs // n_snapshots)

        for epoch in range(self.epochs):
            activations, zs = self.forward(X)
            self.backward(activations, zs, y)
            
            y_pred = activations[-1].flatten()
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            
            preds = (y_pred > 0.5).astype(int)
            accuracy = np.mean(preds == y)
            
            entry = {
                "epoch": epoch + 1,
                "loss": float(loss),
                "accuracy": float(accuracy)
            }

            # Boundary snapshot at regular intervals
            if grid_X is not None and (epoch % snapshot_interval == 0 or epoch == self.epochs - 1):
                entry["boundary"] = self.predict(grid_X)

            self.history.append(entry)
            
        return self.history

    def predict(self, X):
        activations, _ = self.forward(X)
        y_pred = activations[-1].flatten()
        return (y_pred > 0.5).astype(int).tolist()
