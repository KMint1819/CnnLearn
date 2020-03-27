from .layer import Layer


class Model(Layer):
    '''%
    '''

    def __init__(self):
        '''%
        '''
        super().__init__()
        self._layer_list = []

    def __call__(self, data):
        '''%
        '''
        for layer in self._layer_list:
            data = layer(data)
        return data

    def add(self, layer):
        '''%
        '''
        self._layer_list.append(layer)
        return self._layer_list
