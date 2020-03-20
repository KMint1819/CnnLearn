import numpy as np
from .layer import Layer


class Conv2D(Layer):
    def __init__(self, kernel_size=3, zero_pad='same', kernel_list=[]):
        '''%
        '''
        super().__init__()
        assert type(kernel_size) == int
        assert type(zero_pad) == str
        self.kernel_size = kernel_size
        self.zero_pad = zero_pad.lower()
        self.kernel_list = kernel_list
        

    @staticmethod
    def _conv(data1: np.ndarray, data2: np.ndarray):
        '''Do the actual convolution for a filter without sliding
        '''
        assert data1.shape == data2.shape
        result = 0
        for i in range(data1.shape[0]):
            for j in range(data1.shape[1]):
                result += data1[i][j] * data2[i][j]
        return result

    def __call__(self, data: np.ndarray):
        '''%
        '''
        assert data.ndim == 2
        p = None
        if self.zero_pad.lower() == 'same':
            p = (self.kernel_size - 1)/2
        elif self.zero_pad.lower() == 'valid':
            p = 0
        else:
            raise IndexError('zero_pad should be either "same" or "valid"')
        data = np.pad(data, int(p), 'constant')
        rolnum = data.shape[0] - self.kernel_size + 1
        colnum = data.shape[1] - self.kernel_size + 1

        result = np.ndarray((rolnum, colnum))

        for i in range(rolnum):
            for j in range(colnum):
                arr = np.ndarray((self.kernel_size, self.kernel_size))
                for r in range(self.kernel_size):
                    for c in range(self.kernel_size):
                        arr[r][c] = data[i + r][j + c]
                result[i][j] = self._conv(arr, self.kernel_list[0])
        return result
