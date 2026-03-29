import numpy as np
from core.decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.history = []

    def fit(self, X, y, grid_X=None):
        m, n = X.shape
        for i in range(self.n_estimators):
            idx = np.random.choice(m, m, replace=True)
            X_sample, y_sample = X[idx], y[idx]
            
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

            y_pred = self.predict(X)
            accuracy = np.mean(y_pred == y)

            entry = {
                "epoch": i + 1,
                "loss": float(max(0, 1.0 - accuracy)),
                "accuracy": float(accuracy)
            }

            # Boundary snapshot after each tree is added
            if grid_X is not None:
                entry["boundary"] = self.predict(grid_X)

            self.history.append(entry)
            
        return self.history

    def predict(self, X):
        if not self.trees: return [0] * len(X)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.round(np.mean(tree_preds, axis=0)).astype(int).tolist()
