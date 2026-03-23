import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.history = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1
        X_s = (X - mean) / std

        for epoch in range(self.epochs):
            z = np.dot(X_s, self.weights) + self.bias
            y_pred = self._sigmoid(z)
            
            dw = (1 / n_samples) * np.dot(X_s.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            w_orig = self.weights / std
            b_orig = self.bias - np.sum(self.weights * mean / std)
            
            loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            acc = np.mean((y_pred > 0.5).astype(int) == y)
            
            self.history.append({
                "epoch": epoch + 1,
                "loss": float(loss),
                "accuracy": float(acc),
                "weights": w_orig.tolist(),
                "bias": float(b_orig)
            })

        return self.history

    def predict(self, X):
        return (self._sigmoid(np.dot(X, self.weights) + self.bias) > 0.5).astype(int).tolist()
