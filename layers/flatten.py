import numpy as np

from .layer import Layer


class Flatten(Layer):
    '''%
    '''

    def __init__(self):
        '''%
        '''
        pass

    def forward(self, data: np.ndarray):
        '''%
        '''
        return data.flatten()

    def __call__(self, data: np.ndarray):
        '''%
        '''
        return self.forward(data)
