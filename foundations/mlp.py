import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        
        '''
        '''

        relu = lambda z: np.maximum(0, z)
        
        num_layers = len(weights)
        assert num_layers == len(biases)

        a = x.copy()
        for i in range(num_layers):
            a = a @ weights[i] + biases[i]
            if i == num_layers - 1:
                break
            a = relu(a)

        return np.round(a, 5)