import numpy as np
from .layer import Layer


class Conv2D(Layer):
    def __init__(self, kernel_size=3, zero_pad='same', stride=1, kernel_list=[]):
        """
        Keyword Arguments:
            kernel_size {int} -- [Size of kernel] (default: {3})
            zero_pad {str} -- [Type of padding. Choices are 'valid', 'same'] (default: {'same'})
            kernel_list {list} -- [Kernel to convolve.] (default: {[]})
        """
        super().__init__()
        assert type(kernel_size) == int and kernel_size > 0
        assert type(zero_pad) == str and zero_pad in ('same', 'valid')
        assert type(stride) == int and stride > 0
        self.kernel_size = kernel_size
        self.zero_pad = zero_pad.lower()
        self.stride=stride
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
        rolnum = int((data.shape[0] - self.kernel_size)/self.stride + 1)
        colnum = int((data.shape[1] - self.kernel_size)/self.stride + 1)
        print(f'Number of output: {rolnum} * {colnum} = {rolnum * colnum}')
        result = np.ndarray((rolnum, colnum))

        for i in range(rolnum):
            for j in range(colnum):
                # print(f'Convolving for up-left({i}, {j}), down-right({i + self.kernel_size},{j + self.kernel_size})')
                arr = np.ndarray((self.kernel_size, self.kernel_size))
                x = i * self.stride
                y = j * self.stride
                for r in range(self.kernel_size):
                    for c in range(self.kernel_size):
                        arr[r][c] = data[x + r][y + c]
                result[i][j] = self._conv(arr, self.kernel_list[0])
        return result
