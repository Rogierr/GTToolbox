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

    FD_upper = 1 - 0.25*x[1]
    FD_bottom = 1 - 0.25*2*x[0] - 2/3*x[1]

    upper_result = np.min(np.multiply(res_p1, FD_upper))
    bottom_result = np.min(np.multiply(res_p1, FD_bottom))

    return np.max([upper_result, bottom_result])

def payoff_p2_with_FD(x):

    res_p2 = np.dot(x, payoffs_p2)

    FD_left = 1 - 0.25*2*x[1]
    FD_right = 1 - 0.25*x[0] - 2/3*x[1]

    # print("Left results", FD_left)
    # print("Right results", FD_right)
    # print("Results p2", res_p2)

    left_result = np.min(np.multiply(res_p2, FD_left))
    right_result = np.min(np.multiply(res_p2, FD_right))

    return np.max([left_result, right_result])

def maximin_p1_FD(x):

    res_p1 = -np.dot(x, payoffs_p1)

    FD_left = 1 - 0.25 * 2 * x[1]
    FD_right = 1 - 0.25 * x[0] - 2 / 3 * x[1]

    left_result = -np.max(np.multiply(res_p1, FD_left))
    right_result = -np.max(np.multiply(res_p1, FD_right))

    return np.min([left_result, right_result])

def maximin_p2_FD(x):

    res_p2 = -np.dot(payoffs_p2, x)

    FD_upper = 1 - 0.25*x[1]
    FD_bottom = 1 - 0.25*2*x[0] - 2/3*x[1]

    upper_result = -np.max(np.multiply(res_p2, FD_upper))
    bottom_result = -np.max(np.multiply(res_p2, FD_bottom))

    return np.min([upper_result, bottom_result])

def threat_point_p1(x):
    "Function in order to determinate the threat point for p1"

    return np.max(np.dot(payoffs_p1, x))


def threat_point_p2(x):

    return np.max(np.dot(x, payoffs_p2))

def maximin_p1(x):
    "Function in order to determine the maximin strategy for p1"

    return np.max(-np.dot(x, payoffs_p1))

def maximin_p2(x):
    "Function in order to determine the maximin strategy for p2"

    return np.max(-np.dot(payoffs_p2, x))

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

maximini_p1 = minimize(maximin_p1, x_p2, bounds=bnds, constraints=con, options={'disp': True})

print("Player 1 wants to maximize his own payoff first and plays:", maximini_p1.x)
print("The maximin result found for Player 1:", -maximini_p1.fun)

maximini_p2 = minimize(maximin_p2, x_p1, bounds=bnds, constraints=con, options={'disp': True})

print("Player 2 wants to maximize his own payoff first and plays:", maximini_p2.x)
print("The maximin result found for Player 2:", -maximini_p2.fun)

# tp_p1_with_FD = minimize(payoff_p1_with_FD, x_p1, bounds=bnds, constraints=con, options={'disp': False})
#
# print("Player 2 wants to minimize the payoff of Player 1 and plays:", tp_p1_with_FD.x)
# print("The threat point of Player 1:", tp_p1_with_FD.fun)
#
# maxmin_p1_with_FD = minimize(maximin_p1_FD, x_p2, bounds=bnds, constraints=con, options={'disp': False})
#
# print("Player 1 wants to maximize his payoff and plays:", maxmin_p1_with_FD.x)
# print("Maximin result of Player 1:", maxmin_p1_with_FD.fun)
#
# tp_p2_with_FD = minimize(payoff_p2_with_FD, x_p2, bounds=bnds, constraints=con, options={'disp': True})
#
# maxmin_p2_with_FD = minimize(maximin_p2_FD, x_p1, bounds=bnds, constraints=con, options={'disp': False})
#
# print("Player 2 wants to maximize his payoff and plays:", maxmin_p2_with_FD.x)
# print("Maximin result of Player 2:", maxmin_p2_with_FD.fun)