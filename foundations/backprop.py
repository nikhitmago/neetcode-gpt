import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)

        '''
        z = wx + b
        yh = 1 / (1 + e^-z)
        L = 1/2 * [ (yh - yt) ^ 2 ]

        uber:
        dL/dw
        dL/db

        dL/dyh = 2 * (yh - yt)

        dyh/dz = yh * (1 - yh)

        dz/dw = x
        dz/db = 1

        dL/dw = dL/dyh * dyh/dz * dz/dw = 2 * (yh - yt) * yh * (1 - yh) * x
        dL/db = dL/dyh * dyh/dz * dz/db = 2 * (yh - yt) * yh * (1 - yh) * 1
        
        '''
        
        def sigmoid(val):
            return 1 / (1 + np.exp(-val))

        # Forward pass
        z = np.dot(w, x) + b
        yh = sigmoid(z)

        # Loss
        L = 0.5 * (np.square(yh - y_true))

        # Interim gradients
        dL_dyh = yh - y_true
        dyh_dz = yh * (1 - yh)
        dz_dw = x
        dz_db = 1.0

        dL_dw = dL_dyh * dyh_dz * dz_dw
        dL_db = dL_dyh * dyh_dz * dz_db

        return (np.round(dL_dw, 5), round(dL_db, 5))