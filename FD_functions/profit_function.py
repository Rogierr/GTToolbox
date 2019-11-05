import numpy as np
import sympy as symp
from scipy.optimize import minimize_scalar

def profit_function(mu):

    # x = symp.Symbol('x')
    #
    # profit = float(1/3.75) * ((4 + (0.75/mu**2)) - (12 + ((1-12*mu**2.5)/mu**1.5)))

    # profit_test = lambda x: float(1/3.75) * ((4 + (0.75/x**2)) - (12 + ((1-12*x**2.5)/x**1.5)))
    # global_min_test = minimize_scalar(profit_test, bounds=(0, 1), method='bounded')
    #
    # deriv_func = symp.diff(float(1/3.75) * ((4 + (0.75/x**2)) - (12 + ((1-12*x**2.5)/x**1.5))))
    # print(symp.solve(deriv_func))

    profit = mu

    return profit
