import scipy.optimize
from scipy.optimize import linprog
import numpy as np

# payoff_p1 = np.array([[10, 3], [5, 9]])
# print(payoff_p1)

A = np.array([[-4, -2], [-3, -5]])
b = np.array([-1, -1])
c = np.array([1, 1])

x0_bounds = (0, None)
x1_bounds = (0, None)

bounds = [x0_bounds, x1_bounds]

tp = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
print(tp)