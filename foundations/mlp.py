import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        
        for i in range(len(weights) - 1):
            x = np.maximum(0, x @ weights[i] + biases[i])

        output = x @ np.array(weights[-1]) + np.array(biases[-1])

        return np.round(output, 5)