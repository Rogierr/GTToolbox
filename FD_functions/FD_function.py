def FD_function(self, x):
    FD = 1 - 0.25 * (x[:, 1] + x[:, 2]) - (1 / 3) * x[:, 3] - (1 / 2) * (x[:, 5] + x[:, 6]) - (2 / 3) * x[:, 7]

    return FD