import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: [1, 0, 1] true labels (0 or 1)
        # y_pred: [0.9, 0.1, 0.8] predicted probabilities
        
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        
        eps = 1e-7
        y_pred += eps
        
        loss = -1 * np.mean(
            y_true * np.log(y_pred) + \
            (1 - y_true) * np.log(1 - y_pred)
        )
        return round(loss, 4)
        

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        '''
        y_true = [
            [1, 0, 1],
            [0, 0, 1]
        ]

        y_pred = [
            [0.6, 0.4, 0.3],
            [0.3, 0.8, 0.5]
        ]

        ce = -mean(plogq)
        '''
        
        eps = 1e-7
        y_pred += eps

        loss = -1 * np.mean(
            np.sum(
                y_true * np.log(y_pred),
                axis=1
            )
        )
        return round(loss, 4)