import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        
        '''
        X.shape = (3,1)
        y.shape = (3,)
        w.shape = (1,1)
        '''
        N, D = X.shape
        W = np.zeros((D,))
        b = 0

        for _ in range(epochs):
            ## Forward
            y_hat = X @ W + b  # (N,D) @ (D,) = (N,)
            
            ## Loss (not really required)
            L = np.mean(np.square(y_hat - y))

            ## Backward
            dL_dW = (2.0/N) * ((y_hat - y) @ X)  # (N,) @ (N,D) = (D,)
            dL_db = (2.0/N) * np.sum(y_hat - y)

            ## Optimizer step
            W -= lr * dL_dW
            b -= lr * dL_db
        
        return (np.round(W, 5), round(b, 5))