import numpy as np
# import tensorflow as tf
from layers.model import Model
from layers.conv2d import Conv2D


def main():
    data = [
        [2, 3, 7, 4, 6, 2, 9],
        [6, 6, 9, 8, 7, 4, 3],
        [3, 4, 8, 3, 8, 9, 7],
        [7, 8, 3, 6, 6, 3, 4],
        [4, 2, 1, 8, 3, 4, 6],
        [3, 2, 4, 1, 9, 8, 3],
        [0, 1, 3, 9, 2, 1, 4]
    ]
    kernel = [
        [3, 4, 4],
        [1, 0, 2],
        [-1, 0, 3]
    ]

    conv1 = Conv2D(
        kernel_size=3,
        zero_pad='valid',
        stride=2,
        kernel_list=[np.array(kernel)])

    print(conv1(np.array(data)))
    
    #
    ans = [
        [91, 100, 83],
        [69, 91, 127],
        [44, 72, 74]
    ]
    print(f'Correct answer:\n{np.array(ans)}')

if __name__ == "__main__":
    main()
