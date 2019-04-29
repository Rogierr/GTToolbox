from FD_functions.profit_function import profit_function

x_test = [1.45699E-05, 1.66982E-07, 1.96353E-05, 8.99676E-05, 7.74149E-07, 0.005472622, 5.68714E-06, 0.994396577]
x = [0, 0, 0, 0, 0, 0, 1, 0]

rho = float(3 / 8) * (x[1] + x[2] + 2 * (x[5] + x[6])) + float(1 / 2) * (x[3] + 2 * x[7])
mu = 1 + (1 - 0.05) * (2 * rho ** 3 - 3 * rho ** 2)
print(rho)
print(mu)
profit = (profit_function(mu))
print(profit)


payoff_p1 = 7
payoff_p2 = 3.5

result_fish_p1 = (mu*payoff_p1)
result_fish_p2 = (mu*payoff_p2)

print("P1 receives:", profit*result_fish_p1)
print("P2 receives:", profit*result_fish_p2)

