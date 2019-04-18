from FD_functions.profit_function import profit_function
from FD_functions.mu_function import mu_function

x = [1.45699E-05, 1.66982E-07, 1.96353E-05, 8.99676E-05, 7.74149E-07, 0.005472622, 5.68714E-06, 0.994396577]
rho = float(3 / 8) * (x[1] + x[2] + 2 * (x[5] + x[6])) + float(1 / 2) * (x[3] + 2 * x[7])
mu = 1 + (1 - 0.05) * (2 * rho ** 3 - 3 * rho ** 2)
print(mu)
print(mu)
print(profit_function(mu))