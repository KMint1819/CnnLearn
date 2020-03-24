import numpy as np
from layers.pool import MaxPool
from layers.model import Model


def main():
    data = [
        [1, 3, 2, 1],
        [2, 9, 1, 1],
        [1, 3, 2, 3],
        [5, 6, 1, 2]
    ]

    pool1 = MaxPool(
        kernel_size=2,
        stride=2
    )
    print(pool1(np.array(data)))
    #
    ans = [
        [9, 2],
        [6, 3]
    ]
    print(f'Correct answer:\n{np.array(ans)}')

if __name__ == "__main__":
    main()
