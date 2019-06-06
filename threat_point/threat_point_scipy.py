from scipy.optimize import minimize
import numpy as np

payoffs_p1 = np.array([[16, 14], [28, 24]])
payoffs_p2 = np.array([[16, 28], [14, 24]])

x_p1 = np.zeros(2)
x_p2 = np.zeros(2)

def sum_con(x):

    return 1 - np.sum(x)

def payoff_p1_with_FD(x):

    res_p1 = np.dot(payoffs_p1, x)
    FD = 1 - 0.25()

def threat_point_p1(x):
    "Function in order to determinate the threat point for p1"

    return np.max(np.dot(payoffs_p1, x))


def threat_point_p2(x):

    return np.max(np.dot(x, payoffs_p2))

bnds = [(0, 1), (0, 1)]
con = [{'type': 'eq', 'fun': sum_con}]

tp_p1 = minimize(threat_point_p1, x_p1, bounds=bnds,
                             constraints=con, options={'disp': True})  # minimizer threat point p1

print("Player 2 wants to minimize payoff Player 1 and plays:", tp_p1.x)
print("Therefore the threat point of Player 1:", tp_p1.fun)

tp_p2 = minimize(threat_point_p2, x_p2, bounds=bnds,
                             constraints=con, options={'disp': True})  # minimizer threat point p1

print("Player 1 wants to minimize payoff Player 2 and plays:", tp_p2.x)
print("Therefore the threat point of Player 2:",tp_p2.fun)