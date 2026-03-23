import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Simple feature scaling
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1
        X_s = (X - mean) / std

        for epoch in range(self.epochs):
            y_pred = np.dot(X_s, self.weights) + self.bias
            
            dw = (1 / n_samples) * np.dot(X_s.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Record in original scale for frontend
            w_orig = self.weights / std
            b_orig = self.bias - np.sum(self.weights * mean / std)
            
            acc = np.mean((y_pred > 0.5).astype(int) == y)
            self.history.append({
                "epoch": epoch + 1,
                "loss": float(np.mean((y - y_pred)**2)),
                "accuracy": float(acc),
                "weights": w_orig.tolist(),
                "bias": float(b_orig)
            })

        self.weights = self.weights / std
        self.bias = self.bias - np.sum(self.weights * std * mean / std)
        return self.history

    def predict(self, X):
        return (np.dot(X, self.weights) + self.bias > 0.5).astype(int).tolist()
