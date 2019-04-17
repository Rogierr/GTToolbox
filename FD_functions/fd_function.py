import numpy as np

def fd_function(x):

    fd = 1 - 0.25 * (x[:, 1] + x[:, 2]) - (1 / 3) * x[:, 3] - (1 / 2) * (x[:, 5] + x[:, 6]) - (2 / 3) * x[:, 7]

    #fd = 1 - 0.25 * (x[:, 1] + x[:, 2]) - (1 / 3) * x[:, 3] - (1 / 2) * (x[:, 5] + x[:, 6]) \
    #     - np.multiply(float(2/3),x[:, 7])
    return fd
