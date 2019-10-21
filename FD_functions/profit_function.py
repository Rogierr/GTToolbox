import numpy as np
from scipy.optimize import minimize_scalar

def profit_function(mu):

    profit = float(1/3.75) * ((4 + (0.75/mu**2)) - (12 + ((1-12*mu**2.5)/mu**1.5)))

    profit_test = lambda x: float(1/3.75) * ((4 + (0.75/x**2)) - (12 + ((1-12*x**2.5)/x**1.5)))
    global_min_test = minimize_scalar(profit_test, bounds=(0, 1), method='bounded')

    print("Global minima found", global_min_test.fun)
    print("With an x of", global_min_test.x)

    return profit
