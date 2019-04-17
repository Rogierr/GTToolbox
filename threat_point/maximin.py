import numpy as np
import time

from computation.balance_equation import balance_equation
from computation.frequency_pairs_p1 import frequency_pairs_p1
from computation.frequency_pairs_p2 import frequency_pairs_p2
from computation.payoffs_sorted import payoffs_sorted
from computation.random_strategy_draw import random_strategy_draw


def optimized_maximin(self, points, show_strat_p1, show_strat_p2, FD_yn):
    "This is an optimized version for determining the maximin result"

    print("Start of the maximin algorithm")

    ## Start of p1 maximin ##

    start_time = time.time()  # START TIME

    # flatten the transition matrices
    flatten1_1 = self.transition_matrix_game1_to1.flatten()
    flatten2_1 = self.transition_matrix_game2_to1.flatten()

    # store and compute some action stuff
    actions_p2_game1 = self.payoff_p1_game1.shape[1]
    actions_p2_game2 = self.payoff_p1_game2.shape[1]
    total_actions_p2 = actions_p2_game1 + actions_p2_game2

    actions_p1_game1 = self.payoff_p1_game1.shape[0]
    actions_p1_game2 = self.payoff_p1_game2.shape[0]
    total_actions_p1 = actions_p1_game1 + actions_p1_game2

    # flatten the payoffs
    payoff_p1_game_1flatten = self.payoff_p1_game1.flatten()
    payoff_p1_game_2flatten = self.payoff_p1_game2.flatten()

    # compute and store some payoffs stuff
    total_payoffs_p1_game1 = payoff_p1_game_1flatten.size
    total_payoffs_p1_game2 = payoff_p1_game_2flatten.size
    total_payoffs_p1 = total_payoffs_p1_game1 + total_payoffs_p1_game2

    payoff_p2_game_1flatten = self.payoff_p2_game1.flatten()
    payoff_p2_game_2flatten = self.payoff_p2_game2.flatten()

    total_payoffs_p2_game1 = payoff_p2_game_1flatten.size
    total_payoffs_p2_game2 = payoff_p2_game_2flatten.size
    total_payoffs_p2 = total_payoffs_p2_game1 + total_payoffs_p2_game2

    total_payoffs_p2_game1 = payoff_p2_game_1flatten.size
    total_payoffs_p2_game2 = payoff_p2_game_2flatten.size
    total_payoffs_p2 = total_payoffs_p2_game1 + total_payoffs_p2_game2

    # initialize the payoff stuff for p1
    payoff_p1 = np.zeros(total_payoffs_p1)
    payoff_p1[0:total_payoffs_p1_game1] = payoff_p1_game_1flatten
    payoff_p1[total_payoffs_p1_game1:total_payoffs_p1] = payoff_p1_game_2flatten

    px = np.concatenate([flatten1_1, flatten2_1], axis=1)  # px for the first time

    y_punisher = random_strategy_draw(points, total_actions_p1)  # draw some strategies

    frequency_pairs = frequency_pairs_p2(self, points, total_actions_p1, total_actions_p2,
                                         y_punisher)  # sort them based on best replies

    # do the balance equations with Aitken's
    frequency_pairs = balance_equation(self, points, actions_p2_game1, actions_p2_game2, total_payoffs_p2_game1,
                                       total_payoffs_p2, frequency_pairs)

    # activate FD
    if self.FD == True:
        if self.FD_function_use == "FD":
            FD = self.FD_function(draw_payoffs)
        elif self.FD_funcion_use == "mu":
            FD = self.mu_function(self.rho_function(draw_payoffs))
    else:
        FD = 1

        # calculate the payoffs
    payoffs = np.sum(np.multiply(frequency_pairs, payoff_p1), axis=1)
    payoffs = np.multiply(FD, payoffs)
    payoffs = payoffs.reshape((payoffs.size, 1))

    max_payoffs = payoffs_sorted(points, payoffs, (actions_p2_game1 * actions_p2_game2))  # sort the payoffs

    nan_delete = np.where(np.isnan(max_payoffs))  # delete results which are NaN (see thesis why)

    max_payoffs = np.delete(max_payoffs, nan_delete[0], 0)  # actually delete these payoffs

    print("")
    print("")
    minimax_found = np.nanmax(np.nanmin(max_payoffs, axis=1))  # look for maximin value
    print("Maximin value for P1 is", minimax_found)
    print("")
    print("")

    if show_strat_p1 == True:
        minimax_indices_p2 = np.where(max_payoffs == minimax_found)
        found_strategy_p2 = y_punisher[minimax_indices_p2[0]]
        fnd_strategy_p2 = found_strategy_p2.flatten()
        fnd_strategy_p2[0:2] = fnd_strategy_p2[0:2] / np.sum(fnd_strategy_p2[0:2])
        fnd_strategy_p2[2:4] = fnd_strategy_p2[2:4] / np.sum(fnd_strategy_p2[2:4])
        print("Player 1 plays stationary strategy:", fnd_strategy_p2)
        print("While player 2 replies with a best pure reply of:", self.best_pure_strategies[minimax_indices_p2[1]])

    end_time = time.time()
    print("Seconds done to generate", points, "points", end_time - start_time)
    print("")
    print("")

    ## End of P1 maximin algorithm ##

    start_time_p2 = time.time()  # start the time

    # flatten the payoffs
    payoff_p2_game_1flatten = self.payoff_p2_game1.flatten()
    payoff_p2_game_2flatten = self.payoff_p2_game2.flatten()

    # compute and store the payoffs
    total_payoffs_p2_game1 = payoff_p2_game_1flatten.size
    total_payoffs_p2_game2 = payoff_p2_game_2flatten.size
    total_payoffs_p2 = total_payoffs_p2_game1 + total_payoffs_p2_game2

    # initialize the payoffs and store them
    payoff_p2 = np.zeros(total_payoffs_p2)
    payoff_p2[0:total_payoffs_p2_game1] = payoff_p2_game_1flatten
    payoff_p2[total_payoffs_p2_game1:total_payoffs_p2] = payoff_p2_game_2flatten

    px = np.concatenate([flatten1_1, flatten2_1], axis=1)  # px store

    x_punisher = random_strategy_draw(points, total_actions_p2)  # generate new random strategies for punsher

    frequency_pairs = frequency_pairs_p1(self, points, x_punisher)  # best reponses p1

    # balance equations with Delta Squared
    frequency_pairs = balance_equation(self, points, actions_p1_game1, actions_p1_game2, total_payoffs_p1_game1,
                                       total_payoffs_p1, frequency_pairs)

    # activate FD function if necessary
    if FD_yn == True:
        FD = 1 - 0.25 * (frequency_pairs[:, 1] + frequency_pairs[:, 2]) - (1 / 3) * frequency_pairs[:, 3] - (
                1 / 2) * (frequency_pairs[:, 5] + frequency_pairs[:, 6]) - (2 / 3) * frequency_pairs[:, 7]
    else:
        FD = 1

    if self.rarity == True:
        payoffs = np.sum(np.multiply(frequency_pairs, payoff_p2), axis=1)
        payoffs = np.multiply(self.revenue_function(FD), payoffs)
        payoffs = payoffs.reshape((payoffs.size, 1))
    else:
        # compute the payoffs with payoffs and FD function
        payoffs = np.sum(np.multiply(frequency_pairs, payoff_p2), axis=1)
        payoffs = np.multiply(FD, payoffs)
        payoffs = payoffs.reshape((payoffs.size, 1))

    max_payoffs = payoffs_sorted(points, payoffs, (actions_p1_game1 * actions_p1_game2))  # sort the payoffs

    nan_delete = np.where(np.isnan(max_payoffs))  # check where there are nan's

    max_payoffs = np.delete(max_payoffs, nan_delete[0], 0)  # delete these nan's

    minimax_found_p2 = np.nanmax(np.nanmin(max_payoffs, axis=1))  # find the maximin value for p2
    print("Maximin value for P2 is", minimax_found_p2)
    print("")
    print("")

    if show_strat_p2 == True:
        maximin_indices_p2 = np.where(max_payoffs == minimax_found_p2)
        found_strategy = x_punisher[maximin_indices_p2[0]]
        fnd_strategy = found_strategy.flatten()
        fnd_strategy[0:2] = fnd_strategy[0:2] / np.sum(fnd_strategy[0:2])
        fnd_strategy[2:4] = fnd_strategy[2:4] / np.sum(fnd_strategy[2:4])
        print("Player 2 plays stationairy strategy:", fnd_strategy)
        print("While player 2 replies with a best pure reply of:", self.best_pure_strategies[maximin_indices_p2[1]])

    end_time_p2 = time.time()  # end the timer
    print("Seconds done to generate", points, "points", end_time_p2 - start_time_p2)
    print("")
    print("")