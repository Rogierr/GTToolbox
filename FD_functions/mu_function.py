import numpy as np
from FD_functions.learning_curves import scurve

def mu_function(self, rho):

    m = self.m

    # mu = 1 + (1 - m) * (2 * rho ** 3 - 3 * rho ** 2)
    # mu = np.amax([rho], axis=0, initial=m)
    # print(mu)

    mu = rho

    return mu


def tanh_mu(phi, x):

    return phi+np.tanh((3*x))


def scurve_mu(phi, x):

    return phi+scurve(x)

def learning_curve_mu(x):

    return 1.1*(1 - (0.1/(x+0.1)))