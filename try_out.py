from FD_functions.profit_function import profit_function
import numpy as np

# x_test = [1.45699E-05, 1.66982E-07, 1.96353E-05, 8.99676E-05, 7.74149E-07, 0.005472622, 5.68714E-06, 0.994396577]
# x = [0, 0, 0, 0, 0, 0, 0, 1]
x = [0.  ,       0.58810069 ,0.    ,     0.   ,      0.    ,     0.41189931,
 0. ,        0.        ]


rho = float(3 / 8) * (x[1] + x[2] + 2 * (x[5] + x[6])) + float(1 / 2) * (x[3] + 2 * x[7])
mu = 1 + (1 - 0.05) * (2 * rho ** 3 - 3 * rho ** 2)
print(rho)
print(mu)
profit = (profit_function(mu))
print(profit)


payoffs_p1 = [16, 14, 28, 26, 4, 3.5, 7, 13]
payoffs_p2 = [16, 28, 14, 26, 4, 7, 3.5, 13]

result_fish_p1 = np.sum(np.multiply(x,payoffs_p1))
result_fish_p2 = np.sum(np.multiply(x,payoffs_p2))

result_fish_p1 = (mu*result_fish_p1)
result_fish_p2 = (mu*result_fish_p2)

print("P1 receives:", profit*result_fish_p1)
print("P2 receives:", profit*result_fish_p2)

p1_x0 = 16
p2_x0 = 16

p1_x1 = 0.56539
p2_x1 = 1.13078

p1_x2 = 1.13078
p2_x2 = 0.56539

p1_x3 = -5.8521
p2_x3 = -5.8521

p1_x4 = 4
p2_x4 = 4

p1_x5 = 0.3917
p2_x5 = 0.7834

p1_x6 = 0.7834
p2_x6 = 0.3917

p1_x7 = 35.213
p2_x7 = 35.213

p1_x3_adj = -1.91705
p2_x3_adj = -1.90421
