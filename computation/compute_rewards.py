import numpy as np

from computation.balance_equation_all import balance_equation_all
from computation.random_strategy_draw import random_strategy_draw
from FD_functions.fd_function import fd_function
from FD_functions.mb_function import mb_function_p1, mb_function_p2
from FD_functions.sb_function import sb_function_p1, sb_function_p2
from FD_functions.mu_function import mu_function, learning_curve_mu
from FD_functions.rho_function import rho_function
from FD_functions.profit_function import profit_function


def compute_rewards(self, points: int, DSD: bool = False, benefit_function: str = ''):

    draw_payoffs = random_strategy_draw(points, self.total_payoffs)

    # Calculate the balance equations if ETP

    if self.type == 'ETPGame':

        draw_payoffs = balance_equation_all(self, points, draw_payoffs)

        # End of balance equations

        fd = 1

        # activate the FD function
        if self.FD or self.rarity:
            if self.FD:
                fd = fd_function(draw_payoffs)
            elif self.rarity:
                fd = mu_function(self, rho_function(draw_payoffs))
                mu_indic = np.where(fd < 0.06)

    payoffs_p1 = np.sum(np.multiply(draw_payoffs, self.payoffs_p1.flatten()), axis=1)
    payoffs_p2 = np.sum(np.multiply(draw_payoffs, self.payoffs_p2.flatten()), axis=1)

    if hasattr(self, 'plotting_rarity'):
        if self.plotting_rarity == "Rarity":
            payoffs_p1 = np.multiply(fd, payoffs_p1)
            payoffs_p2 = np.multiply(fd, payoffs_p2)
            print("Plotting with rarity active")
            payoffs_p1 = np.multiply(profit_function(fd), payoffs_p1)
            payoffs_p2 = np.multiply(profit_function(fd), payoffs_p2)
        elif self.FD or self.rarity:
            print("Normal plotting active")
            payoffs_p1 = np.multiply(fd, payoffs_p1)
            payoffs_p2 = np.multiply(fd, payoffs_p2)

    if DSD:
        if benefit_function == 'mb':
            fd_p1 = learning_curve_mu(mb_function_p1(draw_payoffs))
            fd_p2 = learning_curve_mu(mb_function_p2(draw_payoffs))

            payoffs_p1_fd = np.multiply(fd_p1, payoffs_p1)
            payoffs_p2_fd = np.multiply(fd_p2, payoffs_p2)

            payoffs_p1 = np.add(payoffs_p1_fd, payoffs_p1)
            payoffs_p2 = np.add(payoffs_p2_fd, payoffs_p2)
        elif benefit_function == 'sb':
            fd_p1 = learning_curve_mu(sb_function_p1(draw_payoffs))
            fd_p2 = learning_curve_mu(sb_function_p2(draw_payoffs))

            payoffs_p1_fd = np.multiply(fd_p1, payoffs_p1)
            payoffs_p2_fd = np.multiply(fd_p2, payoffs_p2)

            payoffs_p1 = np.add(payoffs_p1_fd, payoffs_p1)
            payoffs_p2 = np.add(payoffs_p2_fd, payoffs_p2)

    # here below we just randomly throw out some stuff

    if self.type == 'ETPGame':
        payoffs_p1 = np.delete(payoffs_p1, mu_indic[0], 0)
        payoffs_p2 = np.delete(payoffs_p2, mu_indic[0], 0)

    delete_indic = np.where(np.isnan(payoffs_p1))
    payoffs_p1 = np.delete(payoffs_p1, delete_indic[0], 0)
    payoffs_p2 = np.delete(payoffs_p2, delete_indic[0], 0)

    return [payoffs_p1, payoffs_p2]