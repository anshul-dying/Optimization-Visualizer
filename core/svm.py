import numpy as np

class SVM:
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, tol=1e-3, max_iter=100):
        self.C = C
        self.kernel_type = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = None
        self.b = 0
        self.X = None
        self.y = None
        self.history = []

    def _kernel(self, x1, x2):
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        if self.kernel_type == 'linear':
            return np.dot(x1, x2.T)
        elif self.kernel_type == 'poly':
            return (np.dot(x1, x2.T) + self.coef0) ** self.degree
        elif self.kernel_type == 'rbf':
            if self.gamma == 'scale':
                gamma = 1.0 / (x1.shape[-1] * x1.var()) if x1.var() != 0 else 1.0
            else:
                gamma = self.gamma
            sq_dist = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
            return np.exp(-gamma * sq_dist)
        elif self.kernel_type == 'sigmoid':
            gamma = 1.0 / x1.shape[-1] if self.gamma == 'scale' else self.gamma
            return np.tanh(gamma * np.dot(x1, x2.T) + self.coef0)
        return 0

    def fit(self, X, y, grid_X=None):
        y = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)
        self.b = 0

        n_snapshots = min(20, self.max_iter)
        snapshot_interval = max(1, self.max_iter // n_snapshots)

        for iteration in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)
            for j in range(n_samples):
                i = np.random.randint(0, n_samples)
                while i == j:
                    i = np.random.randint(0, n_samples)

                xi, xj = X[i], X[j]
                yi, yj = y[i], y[j]

                ki = self._kernel(xi, xi)
                kj = self._kernel(xj, xj)
                kij = self._kernel(xi, xj)
                eta = 2.0 * kij - ki - kj
                if eta >= 0: continue

                ei = self._decision_function(xi) - yi
                ej = self._decision_function(xj) - yj

                if yi != yj:
                    L = max(0, self.alpha[j] - self.alpha[i])
                    H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                else:
                    L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                    H = min(self.C, self.alpha[i] + self.alpha[j])
                
                if L == H: continue

                self.alpha[j] -= yj * (ei - ej) / eta
                self.alpha[j] = np.clip(self.alpha[j], L, H)
                self.alpha[i] += yi * yj * (alpha_prev[j] - self.alpha[j])

                b1 = self.b - ei - yi * (self.alpha[i] - alpha_prev[i]) * self._kernel(xi, xi) - yj * (self.alpha[j] - alpha_prev[j]) * self._kernel(xi, xj)
                b2 = self.b - ej - yi * (self.alpha[i] - alpha_prev[i]) * self._kernel(xi, xj) - yj * (self.alpha[j] - alpha_prev[j]) * self._kernel(xj, xj)
                
                if 0 < self.alpha[i] < self.C: self.b = b1
                elif 0 < self.alpha[j] < self.C: self.b = b2
                else: self.b = (b1 + b2) / 2.0

            y_pred = self.predict(X)
            accuracy = np.mean(y_pred == np.where(y <= 0, 0, 1))
            loss = 0.5 * np.dot(self.alpha, y)
            
            history_entry = {
                "epoch": iteration + 1,
                "loss": float(abs(loss)),
                "accuracy": float(accuracy)
            }
            
            if self.kernel_type == 'linear':
                w_data = self.get_weights_linear()
                if w_data:
                    history_entry["weights"] = w_data[0]
                    history_entry["bias"] = w_data[1]

            # Support vector indices (alpha > threshold)
            sv_indices = np.where(self.alpha > 1e-5)[0]
            history_entry["support_vectors"] = sv_indices.tolist()

            # Boundary snapshot at regular intervals
            if grid_X is not None and (iteration % snapshot_interval == 0 or iteration == self.max_iter - 1):
                boundary_preds = self._predict_batch(grid_X)
                history_entry["boundary"] = boundary_preds
            
            self.history.append(history_entry)

            if np.linalg.norm(self.alpha - alpha_prev) < self.tol:
                # Add final boundary snapshot on convergence
                if grid_X is not None and "boundary" not in history_entry:
                    self.history[-1]["boundary"] = self._predict_batch(grid_X)
                break

        return self.history

    def _decision_function(self, x):
        K = self._kernel(self.X, x)  # with atleast_2d, returns (n_train, 1) for single x
        result = np.sum(self.alpha * self.y * K.flatten()) + self.b
        return float(result)

    def _decision_function_batch(self, X):
        try:
            K = self._kernel(self.X, X)  # (n_train, n_test)
            coeffs = self.alpha * self.y  # (n_train,)
            if K.ndim == 2:
                return coeffs @ K + self.b  # (n_test,)
            else:
                return np.sum(coeffs * K) + self.b
        except Exception:
            # Fallback: per-sample prediction
            return np.array([self._decision_function(x) for x in X])

    def _predict_batch(self, X):
        try:
            raw = self._decision_function_batch(X)
            return (np.array(raw) >= 0).astype(int).tolist()
        except Exception:
            return [1 if self._decision_function(x) >= 0 else 0 for x in X]

    def get_weights_linear(self):
        if self.kernel_type != 'linear':
            return None
        w = np.sum((self.alpha * self.y)[:, np.newaxis] * self.X, axis=0)
        return w.tolist(), float(self.b)

    def predict(self, X):
        preds = []
        for x in X:
            preds.append(1 if self._decision_function(x) >= 0 else 0)
        return np.array(preds).tolist()
