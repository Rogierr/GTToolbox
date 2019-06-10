from scipy.optimize import minimize
import numpy as np

payoffs_p1 = np.array([[16, 14], [28, 24]])
payoffs_p2 = np.array([[16, 28], [14, 24]])

payoffs_p1_s2 = np.array([[4, 3.5], [7, 6]])
payoffs_p2_s2 = np.array([[4, 7], [3.5, 6]])

trans_prob_s1 = np.array([[0.8, 0.7], [0.7, 0.6]])
trans_prob_s2 = np.array([[0.5, 0.4], [0.4, 0.15]])

x_typei = np.zeros(2)

def sum_con(x):

    return 1 - np.sum(x)

# Here below we have the functions for Type I CTP-CSP games

def threat_point_p1(x):
    "Function in order to determinate the threat point for p1"

    return np.max(np.dot(payoffs_p1, x))


def threat_point_p2(x):

    return np.max(np.dot(x, payoffs_p2))


def maximin_p1(x):
    "Function in order to determine the maximin strategy for p1"

    return np.max(np.dot(x, -payoffs_p1))


def maximin_p2(x):
    "Function in order to determine the maximin strategy for p2"

    return np.max(np.dot(-payoffs_p2, x))

# Here below are the functions for the Type I CTP-ESP games

def payoff_p1_with_FD(x):

    res_p1 = np.dot(payoffs_p1, x)

    FD_upper = 1 - 0.25*x[1]
    FD_bottom = 1 - 0.25*2*x[0] - 2/3*x[1]

    upper_result = np.multiply(res_p1[0], FD_upper)
    bottom_result = np.multiply(res_p1[1], FD_bottom)

    return np.max([upper_result, bottom_result])

def payoff_p2_with_FD(x):

    res_p2 = np.dot(x, payoffs_p2)

    FD_left = 1 - 0.5*x[1]
    FD_right = 1 - 0.25*x[0] - 2/3*x[1]

    left_result = np.multiply(res_p2[0], FD_left)
    right_result = np.multiply(res_p2[1], FD_right)

    return np.max([left_result, right_result])


def maximin_p1_FD(x):

    res_p1 = np.dot(x, -payoffs_p1)

    FD_left = 1 - 0.25 * 2 * x[1]
    FD_right = 1 - 0.25 * x[0] - 2 / 3 * x[1]

    left_result = np.multiply(FD_left, res_p1[0])
    right_result = np.multiply(FD_right, res_p1[1])

    return np.max([left_result, right_result])


def maximin_p2_FD(x):

    res_p2 = np.dot(-payoffs_p2, x)

    FD_upper = 1 - 0.25*x[1]
    FD_bottom = 1 - 0.25*2*x[0] - 2/3*x[1]

    upper_result = np.multiply(FD_upper, res_p2[0])
    bottom_result = np.multiply(FD_bottom, res_p2[1])

    return np.min([-upper_result, -bottom_result])



def balance_equation_1(x):
    print("Placeholder")


def balance_equation_2(x):
    print("Placeholder")


def balance_equation_3(x):
    print("Placeholder")


def balance_equation_4(x):
    print("Placeholder")

def fd_eq1(x):
    print("Placeholder")

def fd_eq2(x):
    print("Placeholder")

def fd_eq3(x):
    print("Placeholder")

def fd_eq4(x):
    print("Placeholder")


def threat_point_type_ii(x):
    print("Placeholder again")




















bnds = [(0, 1), (0, 1)]
con = [{'type': 'eq', 'fun': sum_con}]

# tp_p1 = minimize(threat_point_p1, x_p1, bounds=bnds,
#                              constraints=con, options={'disp': False})  # minimizer threat point p1
#
# print("Player 2 wants to minimize payoff Player 1 and plays:", tp_p1.x)
# print("Therefore the threat point of Player 1:", tp_p1.fun)
#
# tp_p2 = minimize(threat_point_p2, x_p2, bounds=bnds,
#                              constraints=con, options={'disp': False})  # minimizer threat point p1
#
# print("Player 1 wants to minimize payoff Player 2 and plays:", tp_p2.x)
# print("Therefore the threat point of Player 2:",tp_p2.fun)
#
# maximini_p1 = minimize(maximin_p1, x_p2, bounds=bnds, constraints=con, options={'disp': False})
#
# print("Player 1 wants to maximize his own payoff first and plays:", maximini_p1.x)
# print("The maximin result found for Player 1:", -maximini_p1.fun)
#
# maximini_p2 = minimize(maximin_p2, x_p1, bounds=bnds, constraints=con, options={'disp': False})
#
# print("Player 2 wants to maximize his own payoff first and plays:", maximini_p2.x)
# print("The maximin result found for Player 2:", -maximini_p2.fun)
#
tp_p1_with_FD = minimize(payoff_p1_with_FD, x_typei, bounds=bnds, constraints=con, options={'disp': False,  'eps': 5,
                                                                                             'maxiter': 10000})

print("Player 2 wants to minimize the payoff of Player 1 and plays:", tp_p1_with_FD.x)
print("The threat point of Player 1:", tp_p1_with_FD.fun)

maxmin_p1_with_FD = minimize(maximin_p1_FD, x_typei, bounds=bnds, constraints=con, options={'disp': False, 'eps': 5,
                                                                                             'maxiter': 10000})

print("Player 1 wants to maximize his payoff and plays:", maxmin_p1_with_FD.x)
print("Maximin result of Player 1:", -maxmin_p1_with_FD.fun)

tp_p2_with_FD = minimize(payoff_p2_with_FD, x_typei, bounds=bnds, constraints=con, options={'disp': False, 'eps': 5,
                                                                                             'maxiter': 10000})

print("Player 1 wants to minimize the payoff of Player 2 and plays:", tp_p2_with_FD.x)
print("The threat point of Player 2:", tp_p2_with_FD.fun)


maxmin_p2_with_FD = minimize(maximin_p2_FD, x_typei, bounds=bnds, constraints=con, options={'disp': False,
                                                                                             'eps': 5,
                                                                                             'maxiter': 10000})

print("Player 2 wants to maximize his payoff and plays:", maxmin_p2_with_FD.x)
print("Maximin result of Player 2:", maxmin_p2_with_FD.fun)

