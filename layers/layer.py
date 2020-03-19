import abc


class Layer(abc.ABC):
    '''%
    '''

    def __init__(self):
        '''%
        '''
        pass

    @abc.abstractmethod
    def __call__(self):
        '''%
        '''
        return NotImplementedError
