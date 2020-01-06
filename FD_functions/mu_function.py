import numpy as np

def mu_function(self, rho):

    m = self.m

    # mu = 1 + (1 - m) * (2 * rho ** 3 - 3 * rho ** 2)
    # mu = np.amax([rho], axis=0, initial=m)
    # print(mu)

    mu = rho

    return mu
