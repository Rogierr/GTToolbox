import numpy as np
import time

from computation.balance_equation import balance_equation
from computation.frequency_pairs_p1 import frequency_pairs_p1
from computation.frequency_pairs_p2 import frequency_pairs_p2
from computation.payoffs_sorted import payoffs_sorted
from computation.random_strategy_draw import random_strategy_draw

from FD_functions.fd_function import fd_function
from FD_functions.rho_function import rho_function
from FD_functions.mu_function import mu_function, tanh_mu, scurve_mu
from FD_functions.profit_function import profit_function
from FD_functions.mb_function import mb_function_p1, mb_function_p2
from FD_functions.sb_function import sb_function_p1, sb_function_p2

__all__ = ['optimized_maximin']


def optimized_maximin(game, points, show_strat_p1, show_strat_p2):
    """This is an optimized version for determining the maximin result"""

    print("Start of the maximin algorithm")

    # Start of p1 maximin

    start_time = time.time()  # START TIME

    y_punisher = random_strategy_draw(points, game.payoff_p1_actions)  # draw some strategies

    print("Drawn strategies for the one punishing: \n", y_punisher)

    frequency_pairs = frequency_pairs_p2(game, points, game.payoff_p1_actions, game.payoff_p2_actions,
                                         y_punisher)  # sort them based on best replies

    print("Sorted strategies based on best reply punishing player: \n", frequency_pairs)

    if game.class_games == 'ETP':
    # do the balance equations with Aitken's
        frequency_pairs = balance_equation(game, points, game.payoff_p2_game1.shape[1], game.payoff_p2_game2.shape[1],
                                       game.payoff_p2_game1.size, game.total_payoffs, frequency_pairs)

    fd = 1

    # activate FD
    if game.FD:
        fd = fd_function(frequency_pairs)
    elif game.rarity:
        fd = mu_function(game, rho_function(frequency_pairs))

    if game.learning_curve == 'tanh':
        if game.mu_function == 'sb':
            fd = tanh_mu(game.phi, sb_function_p1(frequency_pairs))
        elif game.mu_function == 'mb':
            fd = tanh_mu(game.phi, mb_function_p1(frequency_pairs))
        elif not game.mu_function == False:
            raise NameError("Not the correct type of mu function provided")

    if game.learning_curve == 'scurve':
        if game.mu_function == 'sb':
            fd = scurve_mu(game.phi, sb_function_p1(frequency_pairs))
        elif game.mu_function == 'mb':
            fd = scurve_mu(game.phi, mb_function_p1(frequency_pairs))
        elif not game.mu_function == False:
            raise NameError("Not the correct type of mu function provided")

    print("Result of the FD function: \n", fd)

        # payoffs are calculated
    payoffs = np.sum(np.multiply(frequency_pairs, game.payoff_p1), axis=1)

    print("Resulting rewards before FD: \n", payoffs)

    if game.rarity:
        print("Plotting with rarity active")
        payoffs = np.multiply(fd, payoffs)
        payoffs = np.multiply(profit_function(fd), payoffs)
        payoffs = payoffs.reshape((payoffs.size, 1))
    else:
        # compute the payoffs with payoffs and FD function
        payoffs = np.multiply(fd, payoffs)
        payoffs = payoffs.reshape((payoffs.size, 1))

    print("Adjusted FD payoffs: \n", payoffs)

    if game.class_games == 'ETP':
        max_payoffs = payoffs_sorted(points, payoffs, (game.payoff_p1_game1.shape[1] * game.payoff_p1_game2.shape[1]))
    else:
        max_payoffs = payoffs_sorted(points, payoffs, game.payoff_p1_actions)
    # sort the payoffs

    print("Sored payoffs based on strategies: \n", max_payoffs)

    nan_delete = np.where(np.isnan(max_payoffs))  # delete results which are NaN (see thesis why)

    max_payoffs = np.delete(max_payoffs, nan_delete[0], 0)  # actually delete these payoffs

    print("")
    print("")
    minimax_found = np.nanmax(np.nanmin(max_payoffs, axis=1))  # look for maximin value
    print("Maximin value for P1 is", minimax_found)
    print("")
    print("")

    if show_strat_p1:
        minimax_indices_p2 = np.where(max_payoffs == minimax_found)
        found_strategy_p2 = y_punisher[minimax_indices_p2[0]]
        fnd_strategy_p2 = found_strategy_p2.flatten()
        fnd_strategy_p2[0:2] = fnd_strategy_p2[0:2] / np.sum(fnd_strategy_p2[0:2])
        fnd_strategy_p2[2:4] = fnd_strategy_p2[2:4] / np.sum(fnd_strategy_p2[2:4])
        print("Player 1 plays stationary strategy:", fnd_strategy_p2)
        print("While player 2 replies with a best pure reply of:", game.best_pure_strategies[minimax_indices_p2[1]])

    end_time = time.time()
    print("Seconds done to generate", points, "points", end_time - start_time)
    print("")
    print("")

    # End of P1 maximin algorithm

    start_time_p2 = time.time()  # start the time

    x_punisher = random_strategy_draw(points, game.payoff_p2_actions)  # generate new random strategies for punisher

    frequency_pairs = frequency_pairs_p1(game, points, x_punisher)  # best responses p1

    if game.class_games == 'ETP':
    # balance equations with Delta Squared
        frequency_pairs = balance_equation(game, points, game.payoff_p1_game1.shape[0], game.payoff_p1_game2.shape[0],
                                           game.payoff_p1_game1.size, game.total_payoffs, frequency_pairs)

    fd = 1

    # activate FD function if necessary
    if game.FD:
        fd = fd_function(frequency_pairs)
    elif game.rarity:
        fd = mu_function(game, rho_function(frequency_pairs))

    if game.learning_curve == 'tanh':
        if game.mu_function == 'sb':
            fd = tanh_mu(game.phi, sb_function_p2(frequency_pairs))
        elif game.mu_function == 'mb':
            fd = tanh_mu(game.phi, mb_function_p2(frequency_pairs))
        elif not game.mu_function == False:
            raise NameError("Not the correct type of mu function provided")

    if game.learning_curve == 'scurve':
        if game.mu_function == 'sb':
            fd = scurve_mu(game.phi, sb_function_p2(frequency_pairs))
        elif game.mu_function == 'mb':
            fd = scurve_mu(game.phi, mb_function_p2(frequency_pairs))
        elif not game.mu_function == False:
            raise NameError("Not the correct type of mu function provided")

        # payoffs are calculated
    payoffs = np.sum(np.multiply(frequency_pairs, game.payoff_p2), axis=1)

    if game.rarity:
        payoffs = np.multiply(fd, payoffs)
        payoffs = np.multiply(profit_function(fd), payoffs)
        payoffs = payoffs.reshape((payoffs.size, 1))
    else:
        # compute the payoffs with payoffs and FD function
        payoffs = np.multiply(fd, payoffs)
        payoffs = payoffs.reshape((payoffs.size, 1))

    if game.class_games == 'ETP':
        max_payoffs = payoffs_sorted(points, payoffs, (game.payoff_p1_game1.shape[0] * game.payoff_p1_game2.shape[0]))
    else:
        max_payoffs = payoffs_sorted(points, payoffs, game.payoff_p2_actions)
    # sort the payoffs

    nan_delete = np.where(np.isnan(max_payoffs))  # check where there are nan's

    max_payoffs = np.delete(max_payoffs, nan_delete[0], 0)  # delete these nan's

    minimax_found_p2 = np.nanmax(np.nanmin(max_payoffs, axis=1))  # find the maxmin value for p2
    print("Maximin value for P2 is", minimax_found_p2)
    print("")
    print("")

    if show_strat_p2:
        maximin_indices_p2 = np.where(max_payoffs == minimax_found_p2)
        found_strategy = x_punisher[maximin_indices_p2[0]]
        fnd_strategy = found_strategy.flatten()
        fnd_strategy[0:2] = fnd_strategy[0:2] / np.sum(fnd_strategy[0:2])
        fnd_strategy[2:4] = fnd_strategy[2:4] / np.sum(fnd_strategy[2:4])
        print("Player 2 plays stationairy strategy:", fnd_strategy)
        print("While player 2 replies with a best pure reply of:", game.best_pure_strategies[maximin_indices_p2[1]])

    end_time_p2 = time.time()  # end the timer
    print("Seconds done to generate", points, "points", end_time_p2 - start_time_p2)
    print("")
    print("")
