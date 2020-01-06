def rho_function(x):

    # rho = float(3 / 8) * (x[:, 1] + x[:, 2] + 2 * (x[:, 5] + x[:, 6])) + float(1 / 2) * (x[:, 3] + 2 * x[:, 7])

    # rho = float(1/2) * ((x[:, 1] + x[:, 2]) + 2 * x[:, 3])

    rho = 1 - 0.5  * (x[:, 1] + x[:, 2]) - 4 * x[:, 3]
    print(rho)

    return rho
