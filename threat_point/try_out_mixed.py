import numpy as np
from computation.random_strategy_draw import random_strategy_draw
from computation.balance_equation_all import balance_equation_all

from FD_functions.fd_function import fd_function
from FD_functions.mu_function import mu_function
from FD_functions.profit_function import profit_function
from FD_functions.rho_function import rho_function


__all__ = ['mixed_strategy_try_out']


def mixed_strategy_try_out(self, points, iterations):
    best_found = None
    best_found_maxmin = None

    for ite in np.arange(0, iterations):
        if ite % 100 == 0:
            print("Currently at iteration:", ite)
        p1_1 = random_strategy_draw(points, 2)
        p1_2 = random_strategy_draw(points, 2)

        p2_1 = random_strategy_draw(1, 2)
        p2_2 = random_strategy_draw(1, 2)

        x = np.zeros((points, 8))

        c = 0

        for i in np.arange(0, 2):
            for j in np.arange(0, 2):
                x[:, c] = np.multiply(p1_1[:, i], p2_1[:, j])
                c += 1

        for i in np.arange(0, 2):
            for j in np.arange(0, 2):
                x[:, c] = np.multiply(p1_2[:, i], p2_2[:, j])
                c += 1

        x = np.divide(x, 2)

        frequency_pairs = balance_equation_all(self, points, x)

        fd = 1

        # activate the FD function
        if self.FD:
            fd = fd_function(frequency_pairs)
        elif self.rarity:
            fd = mu_function(self, rho_function(frequency_pairs))

        payoffs = np.sum(np.multiply(frequency_pairs, self.payoff_p1), axis=1)

        if self.rarity:
            payoffs = np.multiply(fd, payoffs)
            payoffs = np.multiply(profit_function(fd), payoffs)
            payoffs = payoffs.reshape((payoffs.size, 1))
        else:
            # compute the payoffs with payoffs and FD function
            payoffs = np.multiply(fd, payoffs)
            payoffs = payoffs.reshape((payoffs.size, 1))

        nan_delete = np.where(np.isnan(payoffs))  # delete payoffs which are a NaN
        max_payoffs_p1 = np.delete(payoffs, nan_delete[0], 0)  # actually delete them

        threat_candidate_p1 = np.max(max_payoffs_p1)

        if ite == 0:
            best_found = threat_candidate_p1
        elif best_found > threat_candidate_p1:
            best_found = threat_candidate_p1

    for ite in np.arange(0, iterations):
        if ite % 100 == 0:
            print("Currently at iteration:", ite)
        p1_1 = random_strategy_draw(1, 2)
        p1_2 = random_strategy_draw(1, 2)

        p2_1 = random_strategy_draw(points, 2)
        p2_2 = random_strategy_draw(points, 2)

        x = np.zeros((points, 8))

        c = 0

        for i in range(0, 2):
            for j in range(0, 2):
                x[:, c] = np.multiply(p1_1[:, i], p2_1[:, j])
                c += 1

        for i in range(0, 2):
            for j in range(0, 2):
                x[:, c] = np.multiply(p1_2[:, i], p2_2[:, j])
                c += 1

        x = np.divide(x, 2)

        frequency_pairs = balance_equation_all(self, points, x)

        fd = 1

        # activate the FD function
        if self.FD:
            fd = fd_function(frequency_pairs)
        elif self.rarity:
            fd = mu_function(self, rho_function(frequency_pairs))

        payoffs = np.sum(np.multiply(frequency_pairs, self.payoff_p1), axis=1)

        if self.rarity:
            payoffs = np.multiply(fd, payoffs)
            payoffs = np.multiply(profit_function(fd), payoffs)
            payoffs = payoffs.reshape((payoffs.size, 1))
        else:
            # compute the payoffs with payoffs and FD function
            payoffs = np.multiply(fd, payoffs)
            payoffs = payoffs.reshape((payoffs.size, 1))

        nan_delete = np.where(np.isnan(payoffs))  # delete payoffs which are a NaN
        min_payoffs_p1 = np.delete(payoffs, nan_delete[0], 0)  # actually delete them

        maxmin_cand_p1 = np.min(min_payoffs_p1)

        if ite == 0:
            best_found_maxmin = maxmin_cand_p1
        elif best_found_maxmin < maxmin_cand_p1:
            best_found_maxmin = maxmin_cand_p1

    print("Best found threat is", best_found)
    print("Best found maxmin is", best_found_maxmin)
