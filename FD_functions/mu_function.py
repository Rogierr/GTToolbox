def mu_function(self, rho):

    m = self.m
    mu = 1 + (1 - m) * (2 * rho ** 3 - 3 * rho ** 2)

    return mu
