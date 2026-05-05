import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        """
        X: (n_samples, n_features)
        y: (n_samples,)
        """
        n_samples, n_features = X.shape
        
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0
        
        for epoch in range(epochs):
            y_hat = X @ w + b

            error = y_hat - y
            dL_dw = (2 / n_samples) * (X.T @ error)
            dL_db = (2 / n_samples) * np.sum(error)

            w = w - lr * dL_dw
            b = b - lr * dL_db

        return np.round(w, 5), round(b, 5)
