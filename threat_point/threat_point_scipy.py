from scipy.optimize import minimize
import numpy as np

payoffs_p1 = np.array([[16, 14], [28, 24]])
payoffs_p2 = np.array([[16, 28], [14, 24]])

payoffs_p1_s2 = np.array([[4, 3.5], [7, 6]])
payoffs_p2_s2 = np.array([[4, 7], [3.5, 6]])

trans_prob_s1 = np.array([[0.8, 0.7], [0.7, 0.6]])
trans_prob_s2 = np.array([[0.5, 0.4], [0.4, 0.15]])

x_typei = np.random.beta(0.5, 0.5, 2)
x_typei = x_typei / np.sum(x_typei)

x_typeii = np.random.beta(0.5, 0.5, 4)
x_typeii = x_typeii / np.sum(x_typeii)


def sum_con(x: np.array) -> float:

    return 1 - np.sum(x)

# Here below we have the functions for Type I CTP-CSP games


def threat_point_p1(x: np.array) -> float:
    "Function in order to determinate the threat point for p1"

    return np.max(np.dot(payoffs_p1, x))


def threat_point_p2(x: np.array) -> float:
    "Function that determines the threat point for p2"

    return np.max(np.dot(x, payoffs_p2))


def maximin_p1(x: np.array) -> float:
    "Function in order to determine the maximin strategy for p1"

    return np.max(np.dot(x, -payoffs_p1))


def maximin_p2(x: np.array) -> float:
    "Function in order to determine the maximin strategy for p2"

    return np.max(np.dot(-payoffs_p2, x))


# Here below are the functions for the Type I CTP-ESP games


def payoff_p1_with_FD(x: np.array) -> float:
    """
    This function computes the payoff for P1 in case of the usage of an FD function
    :param x: The stationary strategy of player 2
    :return: The threat point result for player 1
    """
    res_p1 = np.dot(payoffs_p1, x)

    FD_upper = 1 - 0.25*x[1]
    FD_bottom = 1 - 0.25*2*x[0] - 2/3*x[1]

    upper_result = np.multiply(res_p1[0], FD_upper)
    bottom_result = np.multiply(res_p1[1], FD_bottom)

    return np.max([upper_result, bottom_result])


def payoff_p2_with_FD(x: np.array) -> float:
    """
    This function computes the payoff for P2 in case of the usage of an FD function
    :param x: The stationary strategy for player 1
    :return: The threat point result for player 2
    """

    res_p2 = np.dot(x, payoffs_p2)

    FD_left = 1 - 0.5*x[1]
    FD_right = 1 - 0.25*x[0] - 2/3*x[1]

    left_result = np.multiply(res_p2[0], FD_left)
    right_result = np.multiply(res_p2[1], FD_right)

    return np.max([left_result, right_result])


def maximin_p1_FD(x: np.array) -> float:

    res_p1_min = np.dot(x, -payoffs_p1)

    FD_left_min = 1 - 0.25 * 2 * x[1]
    FD_right_min = 1 - 0.25 * x[0] - 2 / 3 * x[1]

    left_result = np.multiply(FD_left_min, res_p1_min[0])
    right_result = np.multiply(FD_right_min, res_p1_min[1])

    return np.max([left_result, right_result])


def maximin_p2_FD(x: np.array) -> float:

    res_p2_min = np.dot(-payoffs_p2, x)

    FD_upper_min = 1 - 0.25*x[1]
    FD_bottom_min = 1 - 0.5*x[0] - 2/3*x[1]

    upper_result = np.multiply(FD_upper_min, res_p2_min[0])
    bottom_result = np.multiply(FD_bottom_min, res_p2_min[1])

    return np.min([-upper_result, -bottom_result])

# HERE BELOW ARE SOME NEW TYPE II FUNCTIONS
#
# def balance_equation_1(x: np.array) -> np.array:
#
#     y_s1 = np.zeros(2)
#     y_s2 = np.zeros(2)
#
#     y_s1[0] = x[0] / np.sum(x[0:2])
#     y_s1[1] = x[1] / np.sum(x[0:2])
#
#     y_s2[0] = x[2] / np.sum(x[2:4])
#     y_s2[1] = x[3] / np.sum(x[2:4])
#
#     Q_upper = np.dot(y_s2, trans_prob_s2[0:2, 0])
#     Q_lower = np.dot(y_s1, trans_prob_s1[0:2, 0]) + Q_upper
#
#     Q = np.divide(Q_upper, Q_lower)
#     Other_Q = 1-Q
#
#     x[0:2] = np.multiply(y_s1, Q)
#     x[2:4] = np.multiply(y_s2, Other_Q)
#
#     return x
#
#
# def balance_equation_2(x: np.array) -> np.array:
#     y_s1 = np.zeros(2)
#     y_s2 = np.zeros(2)
#
#     y_s1[0] = x[0] / np.sum(x[0:2])
#     y_s1[1] = x[1] / np.sum(x[0:2])
#
#     y_s2[0] = x[2] / np.sum(x[2:4])
#     y_s2[1] = x[3] / np.sum(x[2:4])
#
#     Q_upper = np.dot(y_s2, trans_prob_s2[0:2, 1])
#     Q_lower = np.dot(y_s1, trans_prob_s1[0:2, 0]) + Q_upper
#
#     Q = np.divide(Q_upper, Q_lower)
#     Other_Q = 1-Q
#
#
#     x[0:2] = np.multiply(y_s1, Q)
#     x[2:4] = np.multiply(y_s2, Other_Q)
#
#     return x
#
#
# def balance_equation_3(x: np.array) -> np.array:
#     y_s1 = np.zeros(2)
#     y_s2 = np.zeros(2)
#
#     y_s1[0] = x[0] / np.sum(x[0:2])
#     y_s1[1] = x[1] / np.sum(x[0:2])
#
#     y_s2[0] = x[2] / np.sum(x[2:4])
#     y_s2[1] = x[3] / np.sum(x[2:4])
#
#     Q_upper = np.dot(y_s2, trans_prob_s2[0:2, 0])
#     Q_lower = np.dot(y_s1, trans_prob_s1[0:2, 1]) + Q_upper
#
#     Q = np.divide(Q_upper, Q_lower)
#     Other_Q = 1-Q
#
#
#     x[0:2] = np.multiply(y_s1, Q)
#     x[2:4] = np.multiply(y_s2, Other_Q)
#
#     return x
#
#
# def balance_equation_4(x: np.array) -> np.array:
#     y_s1 = np.zeros(2)
#     y_s2 = np.zeros(2)
#
#     y_s1[0] = x[0] / np.sum(x[0:2])
#     y_s1[1] = x[1] / np.sum(x[0:2])
#
#     y_s2[0] = x[2] / np.sum(x[2:4])
#     y_s2[1] = x[3] / np.sum(x[2:4])
#
#     Q_upper = np.dot(y_s2, trans_prob_s2[0:2, 1])
#     Q_lower = np.dot(y_s1, trans_prob_s1[0:2, 1]) + Q_upper
#
#     Q = np.divide(Q_upper, Q_lower)
#     Other_Q = 1-Q
#
#
#     x[0:2] = np.multiply(y_s1, Q)
#     x[2:4] = np.multiply(y_s2, Other_Q)
#
#     return x
#
#
# def fd_eq1(x):
#     print("Placeholder")
#
# def fd_eq2(x):
#     print("Placeholder")
#
# def fd_eq3(x):
#     print("Placeholder")
#
# def fd_eq4(x):
#     print("Placeholder")
#
#
# def threat_point_type_ii(x):
#
#     print("initialized x", x)
#     print("Sum of initialized x", np.sum(x))
#     x_uu = balance_equation_1(x)
#
#     res_p2_s1 = np.dot(payoffs_p1, x_uu[0:2])
#     res_p2_s2 = np.dot(payoffs_p1_s2, x_uu[2:4])
#
#     up_up = res_p2_s1[0] + res_p2_s2[0]
#
#     x_ud = balance_equation_2(x)
#
#     res_p2_s1 = np.dot(payoffs_p1, x_ud[0:2])
#     res_p2_s2 = np.dot(payoffs_p1_s2, x_ud[2:4])
#
#     up_down = res_p2_s1[0] + res_p2_s2[1]
#
#     x_du = balance_equation_3(x)
#
#     res_p2_s1 = np.dot(payoffs_p1, x_du[0:2])
#     res_p2_s2 = np.dot(payoffs_p1_s2, x_du[2:4])
#
#     down_up = res_p2_s1[1] + res_p2_s2[0]
#
#     x_dd = balance_equation_4(x)
#
#     res_p2_s1 = np.dot(payoffs_p1, x_dd[0:2])
#     res_p2_s2 = np.dot(payoffs_p1_s2, x_dd[2:4])
#
#     down_down = res_p2_s1[1] + res_p2_s2[1]
#
#     return np.max([up_up, up_down, down_up, down_down])
#


bnds_typei = [(0, 1), (0, 1)]
bnds_typeii = [(0, 1), (0, 1), (0, 1), (0, 1)]

con = [{'type': 'eq', 'fun': sum_con}]

test_n = 1000000

p1_res = np.zeros(test_n)
p2_res = np.zeros(test_n)

p1_maxi = np.zeros(test_n)
p2_maxi = np.zeros(test_n)

condition_test_p1 = np.zeros(test_n)
condition_test_p2 = np.zeros(test_n)


# for i in np.arange(0, test_n):
#     if np.mod(i, 1000) == 0:
#         print("Still alive at", i)

# tp_p1 = minimize(threat_point_p1, x_typei, bounds=bnds_typei,
#                              constraints=con, options={'disp': False})  # minimizer threat point p1

# print("Player 2 wants to minimize payoff Player 1 and plays:", tp_p1.x)
# print("Therefore the threat point of Player 1:", tp_p1.fun)

# tp_p2 = minimize(threat_point_p2, x_typei, bounds=bnds_typei,
#                              constraints=con, options={'disp': False})  # minimizer threat point p1
#
# # print("Player 1 wants to minimize payoff Player 2 and plays:", tp_p2.x)
# # print("Therefore the threat point of Player 2:",tp_p2.fun)
#
# maximini_p1 = minimize(maximin_p1, x_typei, bounds=bnds_typei, constraints=con, options={'disp': False})
# #
# # print("Player 1 wants to maximize his own payoff first and plays:", maximini_p1.x)
# # print("The maximin result found for Player 1:", -maximini_p1.fun)
#
# maximini_p2 = minimize(maximin_p2, x_typei, bounds=bnds_typei, constraints=con, options={'disp': False})

# print("Player 2 wants to maximize his own payoff first and plays:", maximini_p2.x)
# print("The maximin result found for Player 2:", -maximini_p2.fun)

# p1_res[i] = tp_p1.fun
# p2_res[i] = tp_p2.fun
#
# p1_maxi[i] = maximini_p1.fun
# p2_maxi[i] = maximini_p2.fun

# for i in np.arange(0, test_n):
#     if(p1_maxi[i] <= p1_res[i]):
#         condition_test_p1[i] = True
#     else:
#         condition_test_p1[i] = False
#
#     if(p2_maxi[i] <= p2_res[i]):
#         condition_test_p2[i] = True
#     else:
#         condition_test_p2[i] = False
#
# print(np.all(condition_test_p1))
# print(np.all(condition_test_p2))

def tp_p1_typei_fd():

    tp_p1_with_FD = minimize(payoff_p1_with_FD, x_typei, bounds=bnds_typei, constraints=con, options={'disp': False,
                                                                                                      'ftol': 1e-25,
                                                                                                      'eps': 1,
                                                                                                 'maxiter': 10000})
    print("Player 2 wants to minimize the payoff of Player 1 and plays:", tp_p1_with_FD.x)
    print("The threat point of Player 1:", tp_p1_with_FD.fun)


def tp_p2_typei_fd():

    tp_p2_with_FD = minimize(payoff_p2_with_FD, x_typei, bounds=bnds_typei, constraints=con, options={'disp': False,
                                                                                                      'ftol': 1e-25,
                                                                                                      'eps': 1,
                                                                                                      'maxiter': 10000})

    print("Player 1 wants to minimize the payoff of Player 2 and plays:", tp_p2_with_FD.x)
    print("The threat point of Player 2:", tp_p2_with_FD.fun)

def maximin_p1_typei_fd():
    save_values = np.zeros(3)
    x_try_1 = np.array([1, 0])

    maxmin_p1_try1 = minimize(maximin_p1_FD, x_try_1, bounds=bnds_typei, constraints=con, options={'disp': False,
                                                                                                    'ftol': 1e-25,
                                                                                                    'eps': 1,
                                                                                                    'maxiter': 100})

    save_values[0] = maxmin_p1_try1.fun

    x_try_2 = np.array([0, 1])

    maxmin_p1_try2 = minimize(maximin_p1_FD, x_try_2, bounds=bnds_typei, constraints=con, options={'disp': False,
                                                                                                   'ftol': 1e-25,
                                                                                                   'eps': 1,
                                                                                                   'maxiter': 100})

    save_values[1] = maxmin_p1_try2.fun

    maxmin_p1_try3 = minimize(maximin_p1_FD, x_typei, bounds=bnds_typei, constraints=con, options={'disp': False,
                                                                                                    'ftol': 1e-25,
                                                                                                    'eps': 1,
                                                                                                    'maxiter': 10000})

    save_values[2] = maxmin_p1_try3.fun

    found_value = np.max(save_values)
    found_value = -found_value
    print("The maximin value for Player 1 found is", found_value)


def maximin_p2_typei_fd():
    save_values = np.zeros(3)
    x_try_1 = np.array([1, 0])

    maxmin_p2_fd_try1 = minimize(maximin_p2_FD, x_try_1, bounds=bnds_typei, constraints=con, options={'disp': False,
                                                                                                      'ftol': 1e-25,
                                                                                                    'eps': 1,
                                                                                                    'maxiter': 100})

    save_values[0] = maxmin_p2_fd_try1.fun

    x_try_2 = np.array([0, 1])

    maxmin_p2_fd_try2 = minimize(maximin_p2_FD, x_try_2, bounds=bnds_typei, constraints=con, options={'disp': False,
                                                                                                      'ftol': 1e-25,
                                                                                                      'eps': 1,
                                                                                                      'maxiter': 100})

    save_values[1] = maxmin_p2_fd_try2.fun

    maxmin_p2_fd_try3 = minimize(maximin_p2_FD, x_typei, bounds=bnds_typei, constraints=con, options={'disp': False,
                                                                                                      'ftol': 1e-1000,
                                                                                                      'eps': 5,
                                                                                                      'maxiter': 100000})

    save_values[2] = maxmin_p2_fd_try3.fun

    found_value = np.max(save_values)

    print("The maximin value for Player 2 found is", found_value)
    print(save_values)

maximin_p2_typei_fd()

# print("Maximin result of Player 2:", maxmin_p2_with_FD.fun)

# if np.any(p1_maxi <= p1_res):
#     print("P1 check is true")
#
# if np.any(p2_maxi <= p2_res):
#     print("P2 check is true")

# fd_p1 = minimize(threat_point_type_ii, x_typeii, bounds=bnds_typeii, constraints=con, options={'disp': True})
