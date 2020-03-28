from .layer import Layer


class Sequential(Layer):
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
        """Appending another layer or sub-model to the model

        Arguments:
            layer {Sequential or Layer} -- Layer or model to append

        Returns:
            Sequential -- Model after appending the layers
        """
        self._layer_list.append(layer)
        return self._layer_list
