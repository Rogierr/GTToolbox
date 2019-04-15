import numpy as np


def profit_function(mu):
    first_part = np.divide(1, 3.75)
    second_part = 4 + np.multiply(0.75, np.divide(1, np.power(mu, 2)))
    third_part = 12 + np.multiply((1 - 12 * mu ** 2.5), np.divide(1, np.power(mu, 1.5)))

    combined_last = second_part - third_part

    return np.multiply(first_part, combined_last)
