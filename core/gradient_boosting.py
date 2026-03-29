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

    def fit(self, X, y, grid_X=None):
        m = len(y)
        p = np.mean(y)
        self.base_pred = np.log(p / (1 - p)) if 0 < p < 1 else 0.0
        
        f = np.full(m, self.base_pred)

        for i in range(self.n_estimators):
            p_current = self._sigmoid(f)
            residuals = y - p_current
            
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            tree_preds = np.array(tree.predict(X), dtype=float)
            f += self.learning_rate * tree_preds
            
            y_pred = (self._sigmoid(f) > 0.5).astype(int)
            accuracy = np.mean(y_pred == y)
            loss = -np.mean(y * np.log(self._sigmoid(f) + 1e-15) + (1 - y) * np.log(1 - self._sigmoid(f) + 1e-15))
            
            entry = {
                "epoch": i + 1,
                "loss": float(loss),
                "accuracy": float(accuracy)
            }

            # Boundary snapshot after each boosting round
            if grid_X is not None:
                entry["boundary"] = self.predict(grid_X)

            self.history.append(entry)

        return self.history

    def predict(self, X):
        f = np.full(len(X), self.base_pred)
        for tree in self.trees:
            tree_preds = np.array(tree.predict(X), dtype=float)
            f += self.learning_rate * tree_preds
        return (self._sigmoid(f) > 0.5).astype(int).tolist()
