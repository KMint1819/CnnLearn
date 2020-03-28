import numpy as np

from layers.flatten import Flatten
from layers.fullyconnected import FC

def main():
    data = [
        [2, 3, 7],
        [6, 6, 9]
    ]
    data = np.array(data)
    print(data)
    flat = Flatten()
    data = flat(data)
    print(f'{"-" * 50}\nFlattend: \n{data}')
    w = np.arange(3 * data.size).reshape((data.size, 3))
    fc1 = FC(n_output=3, n_input=data.size)
    fc1._set_weights(w)
    print(f'weights: {w}')
    print(f'Result:\n{fc1(data)}')
    print(f'{"-" * 50}\nCorrect answer: {np.array([312, 345, 378])}')

if __name__ == "__main__":
    main()
