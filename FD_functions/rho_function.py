def rho_function(self, x):
    rho = float(3 / 8) * (x[:, 1] + x[:, 2] + 2 * (x[:, 5] + x[:, 6])) + float(1 / 2) * (x[:, 3] + 2 * x[:, 7])

    return rho