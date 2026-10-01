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
        1) z = x.w + b
        2) y_hat = sigmoid(z)
        3) L = 0.5 * (y_hat - y_true) ** 2

        x,w,b -> z -> y_hat -> L

        --- 3 ---
        dL/dy_hat = (y_hat - y_true)

        --- 2 ---
        dL/dz = dL/dy_hat * dy_hat/dz (known from wiki)
              = (y_hat - y_true) * (y_hat * (1 - y_hat))
        
        --- 1 ---
        dL/dw = dL/dz * dz/dw
              = (y_hat - y_true) * (y_hat * (1 - y_hat)) * x
        dL/db = dL/dz * dz/db
              = (y_hat - y_true) * (y_hat * (1 - y_hat)) * 1.0
        '''

        sigmoid = lambda z: 1 / (1 + np.exp(-z))
        
        z = np.dot(x, w) + b
        y_hat = sigmoid(z)
        
        dL_dw = (y_hat - y_true) * (y_hat * (1 - y_hat)) * x
        dL_db = (y_hat - y_true) * (y_hat * (1 - y_hat))

        return np.round(dL_dw, 5), np.round(dL_db, 5)