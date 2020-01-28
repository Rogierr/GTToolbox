import time
import numpy as np

from computation.balance_equation import balance_equation
from computation.frequency_pairs_p1 import frequency_pairs_p1
from computation.frequency_pairs_p2 import frequency_pairs_p2
from computation.payoffs_sorted import payoffs_sorted
from computation.random_strategy_draw import random_strategy_draw

from FD_functions.fd_function import fd_function
from FD_functions.rho_function import rho_function
from FD_functions.profit_function import profit_function
from FD_functions.mu_function import mu_function, tanh_mu, scurve_mu
from FD_functions.mb_function import mb_function_p1, mb_function_p2
from FD_functions.sb_function import sb_function_p1, sb_function_p2


__all__ = ['threat_point_optimized']


def threat_point_optimized(self, points, show_strat_p1, show_strat_p2, print_text):
    """Optimized threat point algorithm for ETP games"""

    if print_text:
        print("The start of the algorithm for finding the threat point")
        print("First let's find the threat point for Player 1")
        print("")

    start_time = time.time()  # timer start

    y_punisher = random_strategy_draw(points, self.payoff_p2_actions)  # draw strategies for the punisher

    frequency_pairs = frequency_pairs_p1(self, points, y_punisher)  # sort based on best reply

    if self.class_games == 'ETP':
    # do the balance equations calculations
        frequency_pairs = balance_equation(self, points, self.payoff_p1_game1.shape[0], self.payoff_p1_game2.shape[0],
                                           self.payoff_p1_game1.size, self.total_payoffs, frequency_pairs)

    fd = 1

    # activate the FD function
    if self.FD:
        fd = fd_function(frequency_pairs)
    elif self.rarity:
        fd = mu_function(self, rho_function(frequency_pairs))

    if self.learning_curve == 'tanh':
        if self.mu_function == 'sb':
            fd = tanh_mu(self.phi, sb_function_p1(frequency_pairs))
        elif self.mu_function == 'mb':
            fd = tanh_mu(self.phi, mb_function_p1(frequency_pairs))
        elif not self.mu_function == False:
            raise NameError("Not the correct type of mu function provided")

    if self.learning_curve == 'scurve':
        if self.mu_function == 'sb':
            fd = scurve_mu(self.phi, sb_function_p1(frequency_pairs))
        elif self.mu_function == 'mb':
            fd = scurve_mu(self.phi, mb_function_p1(frequency_pairs))
        elif not self.mu_function == False:
            raise NameError("Not the correct type of mu function provided")

    payoffs = np.sum(np.multiply(frequency_pairs, self.payoff_p1), axis=1)

    if self.rarity:
        print("Plotting with rarity active")
        payoffs = np.multiply(fd, payoffs)
        payoffs = np.multiply(profit_function(fd), payoffs)
        payoffs = payoffs.reshape((payoffs.size, 1))
    else:
        # compute the payoffs with payoffs and FD function
        payoffs = np.multiply(fd, payoffs)
        payoffs = payoffs.reshape((payoffs.size, 1))

    if self.class_games == 'ETP':
        max_payoffs = payoffs_sorted(points, payoffs, (self.payoff_p1_game1.shape[0] * self.payoff_p1_game2.shape[0]))
    else:
        max_payoffs = payoffs_sorted(points, payoffs, self.payoff_p1_actions)

    # sort the payoffs
    nan_delete = np.where(np.isnan(max_payoffs))  # delete payoffs which are a NaN

    max_payoffs_p1 = np.delete(max_payoffs, nan_delete[0], 0)  # actually delete them
    threat_point_p1 = np.nanmin(np.nanmax(max_payoffs_p1, axis=1))  # determine the threat point

    if print_text:
        print("")
        print("")
        print("Threat point value is", threat_point_p1)
        print("")
        print("")

    if show_strat_p1:
        threat_point_indices_p1 = np.where(max_payoffs_p1 == threat_point_p1)
        found_strategy_p1 = y_punisher[threat_point_indices_p1[0]]
        fnd_strategy_p1 = found_strategy_p1.flatten()
        fnd_strategy_p1[0:2] = fnd_strategy_p1[0:2] / np.sum(fnd_strategy_p1[0:2])
        fnd_strategy_p1[2:4] = fnd_strategy_p1[2:4] / np.sum(fnd_strategy_p1[2:4])
        print("Player 2 plays stationary strategy:", fnd_strategy_p1)
        print("While player 1 replies with a best pure reply of:",
              self.best_pure_strategies[threat_point_indices_p1[1]])

    end_time = time.time()  # stop the time!

    if print_text:
        print("Seconds done to generate", points, "points", end_time - start_time)
        print("")

    # End of algorithm player 1 (checked)

    # Start of algorithm player 2

    if print_text:
        print("")
        print("")
        print("First start the threat point for player 2")
    start_time_p2 = time.time()  # start the time (for p2)

    x_punisher = random_strategy_draw(points, self.payoff_p1_actions)  # draw some awesome strategies

    frequency_pairs = frequency_pairs_p2(self, points, self.payoff_p2_actions, self.payoff_p1_actions,
                                         x_punisher)  # sort them based on best replies

    if self.class_games == 'ETP':
        # do some balance equation accelerator magic
        frequency_pairs = balance_equation(self, points, self.payoff_p2_game1.shape[1], self.payoff_p2_game2.shape[1],
                                           self.payoff_p2_game1.size, self.total_payoffs, frequency_pairs)

    fd = 1

    # activate FD function
    if self.FD:
        fd = fd_function(frequency_pairs)
    elif self.rarity:
        fd = mu_function(self, rho_function(frequency_pairs))

    if self.learning_curve == 'tanh':
        if self.mu_function == 'sb':
            fd = tanh_mu(self.phi, sb_function_p2(frequency_pairs))
        elif self.mu_function == 'mb':
            fd = tanh_mu(self.phi, mb_function_p2(frequency_pairs))
        elif not self.mu_function == False:
            raise NameError("Not the correct type of mu function provided")

    if self.learning_curve == 'scurve':
        if self.mu_function == 'sb':
            fd = scurve_mu(self.phi, sb_function_p2(frequency_pairs))
        elif self.mu_function == 'mb':
            fd = scurve_mu(self.phi, mb_function_p2(frequency_pairs))
        elif not self.mu_function == False:
            raise NameError("Not the correct type of mu function provided")

    # payoffs are calculated
    payoffs = np.sum(np.multiply(frequency_pairs, self.payoff_p2), axis=1)
    if self.rarity:
        payoffs = np.multiply(fd, payoffs)
        payoffs = np.multiply(profit_function(fd), payoffs)
        payoffs = payoffs.reshape((payoffs.size, 1))
    else:
        # compute the payoffs with payoffs and FD function
        payoffs = np.multiply(fd, payoffs)
        payoffs = payoffs.reshape((payoffs.size, 1))

    if self.class_games == 'ETP':
        max_payoffs = payoffs_sorted(points, payoffs, (self.payoff_p2_game1.shape[1] * self.payoff_p2_game2.shape[1]))
    else:
        max_payoffs = payoffs_sorted(points, payoffs, self.payoff_p2_actions)

    # awesome sorting process
    nan_delete = np.where(np.isnan(max_payoffs))  # look for NaN's

    max_payoffs_p2 = np.delete(max_payoffs, nan_delete[0], 0)  # delete them where necessary
    threat_point_p2 = np.nanmin(np.nanmax(max_payoffs_p2, axis=1))  # determine the threat point

    if print_text:
        print("")
        print("")
        print("Threat point value is", threat_point_p2)
        print("")
        print("")

    if show_strat_p2:
        threat_point_indices_p2 = np.where(max_payoffs_p2 == threat_point_p2)
        found_strategy = x_punisher[threat_point_indices_p2[0]]
        fnd_strategy = found_strategy.flatten()
        fnd_strategy[0:2] = fnd_strategy[0:2] / np.sum(fnd_strategy[0:2])
        fnd_strategy[2:4] = fnd_strategy[2:4] / np.sum(fnd_strategy[2:4])
        print("Player 1 plays stationairy strategy:", fnd_strategy)
        print("While player 2 replies with a best pure reply of:",
              self.best_pure_strategies[threat_point_indices_p2[1]])

    end_time_p2 = time.time()  # stop the time

    if print_text:
        print("")
        print("Seconds done to generate", points, "points", end_time_p2 - start_time_p2)
        print("")
        print("")

    self.threat_point = np.zeros(2)
    self.threat_point = [threat_point_p1, threat_point_p2]  # store the threat point!

    return [threat_point_p1, threat_point_p2]
