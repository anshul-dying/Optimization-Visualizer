import numpy as np
from core.decision_tree import DecisionTree

class GradientBoosting:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.base_pred = None
        self.history = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def fit(self, X, y):
        m = len(y)
        # Initial prediction (log-odds)
        p = np.mean(y)
        self.base_pred = np.log(p / (1 - p)) if 0 < p < 1 else 0.0
        
        f = np.full(m, self.base_pred)

        for i in range(self.n_estimators):
            # Gradient (residuals) for binary cross-entropy
            # p = sigmoid(f), residual = y - p
            p_current = self._sigmoid(f)
            residuals = y - p_current
            
            # Fit a decision tree to residuals (using DT as a regressor here)
            # Since my DT is simple, it works mostly for classification, 
            # but we can use it to fit residuals by taking the mean of y in leaves.
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Update f
            f += self.learning_rate * tree.predict(X)
            
            # Metrics
            y_pred = (self._sigmoid(f) > 0.5).astype(int)
            accuracy = np.mean(y_pred == y)
            loss = -np.mean(y * np.log(p_current + 1e-15) + (1 - y) * np.log(1 - p_current + 1e-15))
            
            self.history.append({
                "epoch": i + 1,
                "loss": float(loss),
                "accuracy": float(accuracy)
            })

        return self.history

    def predict(self, X):
        f = np.full(len(X), self.base_pred)
        for tree in self.trees:
            f += self.learning_rate * tree.predict(X)
        return (self._sigmoid(f) > 0.5).astype(int).tolist()
