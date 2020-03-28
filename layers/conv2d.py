import numpy as np
from .layer import Layer


class Conv2D(Layer):
    def __init__(self, n_kernels, kernel_size=3, zero_pad='same', stride=1, kernel_list=None):
        """Convolution layer. Using N,H,W for input
        Keyword Arguments:
            kernel_size {int} -- [Size of kernel] (default: {3})
            zero_pad {str} -- [Type of padding. Choices are 'valid', 'same'] (default: {'same'})
            kernel_list {list} -- [Kernels to convolve.] (default: {None})
        """
        super().__init__()
        assert type(kernel_size) == int and kernel_size > 0
        assert type(zero_pad) == str and zero_pad in ('same', 'valid')
        assert type(stride) == int and stride > 0
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.zero_pad = zero_pad.lower()
        self.stride = stride
        self.kernel_list = kernel_list
        self.input_ch = None

    def forward(self, data: np.ndarray):
        """Forward pass of layer

        Arguments:
            data {np.ndarray} -- Input data

        Raises:
            IndexError: zero_pad should be either "same" or "valid"

        Returns:
            np.ndarray -- Result of forward pass:
        """
        assert data.ndim == 3
        p = None
        if self.zero_pad.lower() == 'same':
            p = (self.kernel_size - 1)/2
        elif self.zero_pad.lower() == 'valid':
            p = 0
        else:
            raise IndexError('zero_pad should be either "same" or "valid"')
        #
        if self.kernel_list == None:
            self.kernel_list = []
            for i in range(self.n_kernels):
                self.kernel_list.append(np.random.rand(
                    data.shape[0], self.kernel_size, self.kernel_size))
        data = np.pad(data, int(p), 'constant')
        # print(data)
        rolnum = int((data.shape[1] - self.kernel_size)/self.stride + 1)
        colnum = int((data.shape[2] - self.kernel_size)/self.stride + 1)
        print(f'Convolving {data}\n kernels:\n {self.kernel_list}')
        print(
            f'Number of output: {self.n_kernels} * {rolnum} * {colnum} = {self.n_kernels * rolnum * colnum}')
        result = np.ndarray((self.n_kernels, rolnum, colnum))

        for i in range(rolnum):
            for j in range(colnum):
                # print(f'Convolving for up-left({i}, {j}), down-right({i + self.kernel_size},{j + self.kernel_size})')
                x = i * self.stride
                y = j * self.stride
                arr = data[:, x:x+self.kernel_size, y:y+self.kernel_size]
                for out_ch, kernel in enumerate(self.kernel_list):
                    # print(f'Convolving: sub_data:\n {arr}\nkernel:\n{kernel}')
                    result[out_ch][i][j] = np.sum(np.multiply(arr, kernel))
                    # print(f'Result: {result[out_ch][i][j]}\n{"-"*50}')
        return result

    def __call__(self, data: np.ndarray):
        '''%
        '''
        return self.forward(data)
