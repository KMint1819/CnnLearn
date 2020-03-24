'''%
'''
import numpy as np

from .layer import Layer


class MaxPool(Layer):
    def __init__(self, kernel_size=3, stride=1):
        assert type(kernel_size) == int and kernel_size > 1
        assert type(stride) == int and stride > 1
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, data: np.ndarray):
        assert type(data) == np.ndarray
        rolnum = int((data.shape[0] - self.kernel_size)/self.stride + 1)
        colnum = int((data.shape[1] - self.kernel_size)/self.stride + 1)
        print(f'Number of output: {rolnum} * {colnum} = {rolnum * colnum}')
        result = np.ndarray((rolnum, colnum))

        for i in range(rolnum):
            for j in range(colnum):
                # print(f'Pooling for up-left({i}, {j}), down-right({i + self.kernel_size},{j + self.kernel_size})')
                x = i * self.stride
                y = j * self.stride
                m = -99999999999
                for r in range(self.kernel_size):
                    for c in range(self.kernel_size):
                        if data[x + r][y + c] > m:
                            m = data[x + r][y + c]
                result[i][j] = m
        return result

    def __call__(self, data: np.ndarray):
        return self.forward(data)
