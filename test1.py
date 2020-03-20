import numpy as np
# import tensorflow as tf
from layers.model import Model
from layers.conv2d import Conv2D


def main():
    data = [
        [3, 0, 1, 2, 7, 4],
        [1, 5, 8, 9, 3, 1],
        [2, 7, 2, 5, 1, 3],
        [0, 1, 3, 1, 7, 8],
        [4, 2, 1, 6, 2, 8],
        [2, 4, 5, 2, 3, 9]
    ]
    kernel = [
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]]

    conv1 = Conv2D(
        kernel_size=3,
        zero_pad='valid',
        kernel_list=[np.array(kernel)])
    print(conv1(np.array(data)))


if __name__ == "__main__":
    main()
