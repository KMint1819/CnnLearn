import numpy as np
import random

from .layer import Layer


class FC(Layer):
    '''%
    '''

    def __init__(self, n_output, n_input=None):
        '''%
        '''
        self.n_output = n_output
        self.n_input = n_input
        # self.use_bias = use_bias
        self.weights = None

    def _set_weights(self, w):
        """For test purpose
        """        
        self.weights = w

    def forward(self, data: np.ndarray):
        '''%
        '''
        assert type(data) == np.ndarray and data.ndim == 1
        if self.n_input == None:
            self.n_input = data.size
            self.weights = np.random.rand(self.n_input, self.n_output)
        elif data.size != self.n_input:
            print(
                f'Shape of fully connected is not consistent. Expected {self.n_input} while getting {data.size}.')
            raise RuntimeError
        print(f'Fully connecting with data:\n{data}\n and weights({self.n_input * self.n_output}):\n{self.weights}')
        result = np.zeros((self.n_output))
        for i in range(self.n_output):
            print(f'Calculating output of {i+1}-th neuron')
            for j, num in enumerate(data):
                print(f'{j}: {self.weights[j, i]} * {num}')
                result[i] += self.weights[j, i] * num
        return result

    def __call__(self, data: np.ndarray):
        '''%
        '''
        return self.forward(data)
