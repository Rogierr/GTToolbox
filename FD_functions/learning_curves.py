import numpy as np


def scurve(x):
    numerator = np.power(x, 3)
    denominator = 3*np.power(x, 2) - 3*x + 1

    return numerator/denominator
