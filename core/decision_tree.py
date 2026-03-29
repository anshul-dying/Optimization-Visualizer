import numpy as np

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.history = []

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def _gini(self, y):
        m = len(y)
        if m == 0: return 0
        p = np.sum(y) / m
        return 1 - p**2 - (1 - p)**2

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= self.min_samples_split:
            return None, None

        best_gini = 1.0
        best_feature, best_threshold = None, None

        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue

                gini_left = self._gini(y[left_idx])
                gini_right = self._gini(y[right_idx])
                gini = (np.sum(left_idx) * gini_left + np.sum(right_idx) * gini_right) / m

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0, max_depth_override=None):
        effective_max_depth = max_depth_override if max_depth_override is not None else self.max_depth
        m, n = X.shape
        num_labels = len(np.unique(y))

        if depth >= effective_max_depth or num_labels == 1 or m < self.min_samples_split:
            leaf_value = np.round(np.mean(y)) if m > 0 else 0
            return self.Node(value=leaf_value)

        feature, threshold = self._best_split(X, y)
        if feature is None:
            leaf_value = np.round(np.mean(y))
            return self.Node(value=leaf_value)

        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1, max_depth_override)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1, max_depth_override)

        return self.Node(feature, threshold, left, right)

    def fit(self, X, y, grid_X=None):
        if grid_X is not None:
            # Animate progressive tree building: depth 1, 2, 3, ..., max_depth
            actual_max_depth = min(self.max_depth, 10)
            for depth in range(1, actual_max_depth + 1):
                temp_root = self._build_tree(X, y, max_depth_override=depth)
                y_pred_data = np.array([self._predict_one(x, temp_root) for x in X])
                accuracy = np.mean(y_pred_data == y)
                gini = self._gini(y) - accuracy  # approximate impurity decrease

                entry = {
                    "epoch": depth,
                    "loss": float(max(0, 1.0 - accuracy)),
                    "accuracy": float(accuracy),
                    "boundary": [int(self._predict_one(x, temp_root)) for x in grid_X]
                }
                self.history.append(entry)

            # Set the actual root to full depth
            self.root = self._build_tree(X, y)
        else:
            self.root = self._build_tree(X, y)
            y_pred = self.predict(X)
            accuracy = np.mean(y_pred == y)
            self.history.append({
                "epoch": 1,
                "loss": 0,
                "accuracy": float(accuracy)
            })
        return self.history

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X]).tolist()
