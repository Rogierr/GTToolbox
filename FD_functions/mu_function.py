def mu_function(self, rho):
    m = self.m
    #         mu = 1 + (1 - m) * ((n2/(n1-n2))*rho**n1 - (n1/(n1-n2))*rho**n2)
    mu = 1 + (1 - m) * (2 * rho ** 3 - 3 * rho ** 2)

    return mu