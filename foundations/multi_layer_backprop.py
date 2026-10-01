import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        '''
        x.shape = (2,)
        W1.shape = (2,2)
        b1.shape = (2,)
        W2.shape = (1,2)
        b2.shape = (1,)
        y_true = (1,)

        Arch: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions -> Loss
        '''

        x = np.array(x)
        W1, W2 = np.array(W1), np.array(W2)
        b1, b2 = np.array(b1), np.array(b2)
        y_true = np.array(y_true)
        
        n = len(y_true) if y_true.ndim > 0 else 1
        relu = lambda z: np.maximum(0, z)

        ## Forward
        z1 = x @ W1.T + b1  # (2,)
        a1 = relu(z1)  # (2,)
        yhat = W2 @ a1 + b2  # (1,)
        
        ## Loss
        L = np.mean(np.square(yhat - y_true))

        ## Backward (Gradients)

        ### Preds
        dL_dyhat = 2.0 * (yhat - y_true) / n  # (1,)

        ### Second layer
        dL_dW2 = dL_dyhat.reshape(1, -1) @ a1.reshape(1, -1)
        dL_db2 = dL_dyhat * 1.0
        dL_da1 = dL_dyhat @ W2  # (2,)

        ### ReLU
        dL_dz1 = dL_da1 * (z1 > 0).astype(float)  # (2,)

        ### First layer
        dL_dW1 = dL_dz1.reshape(-1, 1) @ x.reshape(1, -1)
        dL_db1 = dL_dz1 * 1.0

        return {
            'loss': np.round(L, 4),
            'dW1': np.round(dL_dW1, 4).tolist(),
            'db1': np.round(dL_db1, 4).tolist(),
            'dW2': np.round(dL_dW2, 4).tolist(),
            'db2': np.round(dL_db2, 4).tolist()
        }