def aitken_delta_squared(q1, q2, q3):
    "This is the Aitken's Delta Squared accelerator, a stable version"

    x3_x2 = np.subtract(q3, q2)
    x2_x1 = np.subtract(q2, q1)

    x3_x2_squared = np.power(x3_x2, 2)
    denominator = np.subtract(x3_x2, x2_x1)

    fraction = np.divide(x3_x2_squared, denominator)

    return np.subtract(q3, fraction)