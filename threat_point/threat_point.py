import time
import numpy as np

def threat_point_optimized(self, points, show_strat_p1, show_strat_p2, print_text, FD_yn):
    "OPtimized threat point algorithm for ETP games"

    def random_strategy_draw(points, number_of_actions):
        "This function draws random strategies from a beta distribution, based on the number of points and actions"

        # draw some strategies and normalize them
        strategies_drawn = np.random.beta(0.5, 0.5, (points, number_of_actions))
        strategies_drawn = strategies_drawn / np.sum(strategies_drawn, axis=1).reshape([points, 1])

        return strategies_drawn

    def frequency_pairs_p1(points, p2_actions, p1_actions, strategies_drawn):
        "This function sorts the strategies based on the responses"

        # store the game size
        game_size_1 = self.payoff_p1_game1.size
        game_size_2 = self.payoff_p1_game2.size

        # store the actions of p1 in both game
        p1_actions_game1 = self.payoff_p1_game1.shape[0]
        p1_actions_game2 = self.payoff_p1_game2.shape[0]

        p1_actions_combi = p1_actions_game1 * p1_actions_game2
        p1_action_range = np.arange(p1_actions_combi)

        # initialize frequency pairs
        frequency_pairs = np.zeros((points * (p1_actions_game1 * p1_actions_game2), game_size_1 + game_size_2))

        # set the range for both games
        p1_act_game1_range = np.arange(p1_actions_game1)
        p1_act_game2_range = np.arange(p1_actions_game2)

        # create best response for game 1
        for i in np.nditer(p1_action_range):
            for j in np.nditer(p1_act_game1_range):
                mod_remain = np.mod(i, p1_actions_game1)
                frequency_pairs[i * points:(i + 1) * points, p1_actions_game1 * mod_remain + j] = strategies_drawn[
                                                                                                  :, j]

        # loop for best responses for game 2
        for i in np.nditer(p1_action_range):
            for j in np.nditer(p1_act_game2_range):
                floor_div = np.floor_divide(i, p1_actions_game2)
                frequency_pairs[i * points:(i + 1) * points,
                j + game_size_1 + (p1_actions_game1 * floor_div)] = strategies_drawn[:, p1_actions_game1 + j]
        return frequency_pairs

    def balance_equation(self, tot_act_ut, tot_act_thr, tot_payoffs_game1, tot_payoffs, frequency_pairs):
        "Calculates the result of the balance equations in order to adjust the frequency pairs"

        # store the game sizes
        game_size_1 = self.payoff_p1_game1.size
        game_size_2 = self.payoff_p1_game2.size

        # initialize yi, Q and Q_new
        yi = np.zeros((points * (tot_act_thr * tot_act_ut), game_size_1 + game_size_2))
        Q = np.zeros((1, points * (tot_act_thr * tot_act_ut)))
        Q_new = np.zeros((1, points * (tot_act_thr * tot_act_ut)))

        # compute Yi
        yi[:, 0:tot_payoffs_game1] = frequency_pairs[:, 0:tot_payoffs_game1] / np.sum(
            frequency_pairs[:, 0:tot_payoffs_game1], axis=1).reshape([points * tot_payoffs_game1, 1])
        yi[:, tot_payoffs_game1:tot_payoffs] = frequency_pairs[:, tot_payoffs_game1:tot_payoffs] / np.sum(
            frequency_pairs[:, tot_payoffs_game1:tot_payoffs], axis=1).reshape(
            [points * (tot_payoffs - tot_payoffs_game1), 1])

        index_values = np.arange(points * (tot_act_thr * tot_act_ut))  # create a range of index values

        p1_px_between = np.asarray(px)  # set px
        p1_px = p1_px_between[0]

        if self.hysteresis == True:
            etp_calculation = np.multiply(self.phi, self.etp_matrix)
        else:
            etp_calculation = self.etp_matrix

        # iterate for 35 iterations
        for i in np.arange(35):

            # first iteration, just calculate Q
            if i == 0:
                new_x = p1_px - np.dot(frequency_pairs, etp_calculation)

                upper_part_Q = np.sum(
                    np.multiply(yi[:, tot_payoffs_game1:tot_payoffs], new_x[:, tot_payoffs_game1:tot_payoffs]),
                    axis=1)
                leftdown_part_Q = np.sum(
                    np.multiply(yi[:, 0:tot_payoffs_game1], (1 - new_x[:, 0:tot_payoffs_game1])), axis=1)
                rightdown_part_Q = np.sum(
                    np.multiply(yi[:, tot_payoffs_game1:tot_payoffs], new_x[:, tot_payoffs_game1:tot_payoffs]),
                    axis=1)

                Q_between = upper_part_Q / (leftdown_part_Q + rightdown_part_Q)
                Q = Q_between

                frequency_pairs[:, 0:tot_payoffs_game1] = (np.multiply(Q, yi[:, 0:tot_payoffs_game1]))
                frequency_pairs[:, tot_payoffs_game1:tot_payoffs] = np.multiply((1 - Q), yi[:,
                                                                                         tot_payoffs_game1:tot_payoffs])

            # for stability, calculate until iteration 9 normal Q
            if i > 0 and i < 10:
                new_x = p1_px - np.dot(frequency_pairs, etp_calculation)

                upper_part_Q = np.sum(
                    np.multiply(yi[:, tot_payoffs_game1:tot_payoffs], new_x[:, tot_payoffs_game1:tot_payoffs]),
                    axis=1)
                leftdown_part_Q = np.sum(
                    np.multiply(yi[:, 0:tot_payoffs_game1], (1 - new_x[:, 0:tot_payoffs_game1])), axis=1)
                rightdown_part_Q = np.sum(
                    np.multiply(yi[:, tot_payoffs_game1:tot_payoffs], new_x[:, tot_payoffs_game1:tot_payoffs]),
                    axis=1)

                Q_between = upper_part_Q / (leftdown_part_Q + rightdown_part_Q)
                Q = np.hstack((Q, Q_between))

                frequency_pairs[:, 0:tot_payoffs_game1] = (np.multiply(Q[:, i], yi[:, 0:tot_payoffs_game1]))
                frequency_pairs[:, tot_payoffs_game1:tot_payoffs] = np.multiply((1 - Q[:, i]), yi[:,
                                                                                               tot_payoffs_game1:tot_payoffs])

            # apply Aitken's
            if i == 10:
                Q_new = self.aitken_delta_squared(Q[:, i - 3], Q[:, i - 2], Q[:, i - 1])
                nan_org = np.where(np.isnan(Q_new))  # check whether Nan's occur
                nan_indic = nan_org[0]

                Q_new[nan_indic, :] = Q_between[nan_indic, :]  # replace NaN with last known value

                Q_old = np.copy(Q_new)

                Q = np.hstack((Q, Q_new))

            # and only Aitken's
            if i > 10:
                Q_new[index_values, :] = self.aitken_delta_squared(Q[index_values, i - 3], Q[index_values, i - 2],
                                                                   Q[index_values, i - 1])
                Q_old2 = np.copy(Q_old)
                nan_res = np.where(np.isnan(Q_new, Q_old))  # check for NaN's
                nan_indices = nan_res[0]

                nan_between = np.where(np.in1d(index_values, nan_indices))
                nan_cands = nan_between[0]

                index_values = np.delete(index_values, nan_cands)  # delete NaN's after being returned in last known

                Q_new[nan_indices, :] = Q_old2[nan_indices, :]

                Q = np.hstack((Q, Q_new))

                Q_old = np.copy(Q_new)
                results = np.where(Q[index_values, i - 1] == Q[index_values, i])  # check whether Q converged
                remove_indices = results[0]

                removal_between = np.where(np.in1d(index_values, remove_indices))
                removal_cands = removal_between[0]
                index_values = np.delete(index_values, removal_cands)

        # compute definitive x
        frequency_pairs[:, 0:tot_payoffs_game1] = (np.multiply(Q[:, 34], yi[:, 0:tot_payoffs_game1]))
        frequency_pairs[:, tot_payoffs_game1:tot_payoffs] = np.multiply((1 - Q[:, 34]),
                                                                        yi[:, tot_payoffs_game1:tot_payoffs])

        return frequency_pairs

    def frequency_pairs_p2(points, p2_actions, p1_actions, strategies_drawn):
        "Create frequency pairs for P2 based on best responses"

        # store the size of the games
        game_size_1 = self.payoff_p2_game1.size
        game_size_2 = self.payoff_p2_game2.size

        # store the ranges of the actions of both players
        p1_actions_range = np.arange(p1_actions)
        p2_actions_range = np.arange(p2_actions)

        p2_actions_game1 = self.payoff_p2_game1.shape[1]
        p2_actions_game2 = self.payoff_p2_game2.shape[1]

        p2_actions_combo = p2_actions_game1 * p2_actions_game2
        p2_action_range = np.arange(p2_actions_combo)

        # initialize frequency pairs
        frequency_pairs = np.zeros((points * (p2_actions_game1 * p2_actions_game2), game_size_1 + game_size_2))

        # loop over the first game
        for i in np.nditer(np.arange(p2_actions_game1)):
            for j in np.nditer(p2_action_range):
                modul = np.mod(j, p2_actions_game1)
                frequency_pairs[j * points:(j + 1) * points, p2_actions_game1 * i + modul] = strategies_drawn[:, i]
        # loop over the second game
        for i in np.nditer(np.arange(p2_actions_game2)):
            for j in np.nditer(p2_action_range):
                divide = np.floor_divide(j, p2_actions_game2)
                frequency_pairs[j * points:(j + 1) * points,
                p2_actions_combo + divide + (i * p2_actions_game2)] = strategies_drawn[:, i + p2_actions_game1]

        return frequency_pairs

    def payoffs_sorted(points, payoffs, actions):
        "This function sorts the payoffs in order to prepare the threat point"

        # create ranges for points and actions
        points_range = np.arange(points)
        actions_range = np.arange(actions)

        payoffs_sort = np.zeros((points, actions))  # nitialize the payoffs sort

        # sort the payoffs!
        for x in np.nditer(points_range):
            for i in np.nditer(actions_range):
                payoffs_sort[x, i] = payoffs[points * i + x]

        return payoffs_sort

    if print_text == True:
        print("The start of the algorithm for finding the threat point")
        print("First let's find the threat point for Player 1")

    # flatten the transition matrices
    flatten1_1 = self.transition_matrix_game1_to1.flatten()
    flatten2_1 = self.transition_matrix_game2_to1.flatten()

    #  store the actions for both players
    actions_p2_game1 = self.payoff_p1_game1.shape[1]
    actions_p2_game2 = self.payoff_p1_game2.shape[1]
    total_actions_p2 = actions_p2_game1 + actions_p2_game2

    actions_p1_game1 = self.payoff_p1_game1.shape[0]
    actions_p1_game2 = self.payoff_p1_game2.shape[0]
    total_actions_p1 = actions_p1_game1 + actions_p1_game2

    # Start of algorithm for player 1

    start_time = time.time()  # timer start

    # flatten payoffs game 1 and 2
    payoff_p1_game_1flatten = self.payoff_p1_game1.flatten()
    payoff_p1_game_2flatten = self.payoff_p1_game2.flatten()

    # store size of the payoffs
    total_payoffs_p1_game1 = payoff_p1_game_1flatten.size
    total_payoffs_p1_game2 = payoff_p1_game_2flatten.size
    total_payoffs_p1 = total_payoffs_p1_game1 + total_payoffs_p1_game2

    # initialize and assign payoffs
    payoff_p1 = np.zeros(total_payoffs_p1)
    payoff_p1[0:total_payoffs_p1_game1] = payoff_p1_game_1flatten
    payoff_p1[total_payoffs_p1_game1:total_payoffs_p1] = payoff_p1_game_2flatten

    px = np.concatenate([flatten1_1, flatten2_1], axis=1)  # store px

    y_punisher = random_strategy_draw(points, total_actions_p2)  # draw strategies for the punisher

    frequency_pairs = frequency_pairs_p1(points, total_actions_p2, total_actions_p1,
                                         y_punisher)  # sort based on best reply

    # do the balance equations calculations
    frequency_pairs = balance_equation(self, actions_p1_game1, actions_p1_game2, total_payoffs_p1_game1,
                                       total_payoffs_p1, frequency_pairs)

    # activate the FD function
    if self.FD == True:
        if self.FD_function_use == "FD":
            FD = self.FD_function(draw_payoffs)
        elif self.FD_function_use == "mu":
            FD = self.mu_function(self.rho_function(draw_payoffs))
    else:
        FD = 1

    payoffs = np.sum(np.multiply(frequency_pairs, payoff_p1), axis=1)

    if self.rarity == True:
        print("Plotting with rarity active")
        rho = self.rho_function(frequency_pairs)
        mu = self.mu_function(rho)
        payoffs = np.multiply(self.profit_function(mu), payoffs)
    else:
        # compute the payoffs with payoffs and FD function
        payoffs = np.multiply(FD, payoffs)
        payoffs = payoffs.reshape((payoffs.size, 1))

    max_payoffs = payoffs_sorted(points, payoffs, (actions_p1_game1 * actions_p1_game2))  # sort the payoffs
    nan_delete = np.where(np.isnan(max_payoffs))  # delete payoffs which are a NaN

    max_payoffs_p1 = np.delete(max_payoffs, nan_delete[0], 0)  # actually delete them
    threat_point_p1 = np.nanmin(np.nanmax(max_payoffs_p1, axis=1))  # determine the threat point

    location_threat = np.where(threat_point_p1 == max_payoffs_p1)

    #         if self.rarity == True:
    #             select_payoffs = np.multiply(self.profit_function(FD),select_payoffs)
    #             select_payoffs = select_payoffs.reshape((select_payoffs.size,1))

    #             second_sort = payoffs_sorted(points,select_payoffs,(actions_p1_game1*actions_p1_game2))

    #             second_sort_p1 = np.delete(second_sort,nan_delete[0],0)
    #             location_x = location_threat[0]
    #             location_y = location_threat[1]
    #             threat_point_p1 = (second_sort_p1[location_x,location_y])[0]

    if print_text == True:
        print("")
        print("")
        print("Threat point value is", threat_point_p1)
        print("")
        print("")

    if show_strat_p1 == True:
        threat_point_indices_p1 = np.where(max_payoffs_p1 == threat_point_p1)
        found_strategy_p1 = y_punisher[threat_point_indices_p1[0]]
        fnd_strategy_p1 = found_strategy_p1.flatten()
        fnd_strategy_p1[0:2] = fnd_strategy_p1[0:2] / np.sum(fnd_strategy_p1[0:2])
        fnd_strategy_p1[2:4] = fnd_strategy_p1[2:4] / np.sum(fnd_strategy_p1[2:4])
        print("Player 2 plays stationary strategy:", fnd_strategy_p1)
        print("While player 1 replies with a best pure reply of:",
              self.best_pure_strategies[threat_point_indices_p1[1]])

    end_time = time.time()  # stop the time!

    if print_text == True:
        print("Seconds done to generate", points, "points", end_time - start_time)
        print("")

    # End of algorithm player 1

    # Start of algorithm player 2

    if print_text == True:
        print("")
        print("")
        print("First start the threat point for player 2")
    start_time_p2 = time.time()  # start the time (for p2)

    # flatten the payoffs of the gamew
    payoff_p2_game_1flatten = self.payoff_p2_game1.flatten()
    payoff_p2_game_2flatten = self.payoff_p2_game2.flatten()

    # check the sizes of the total payoffs
    total_payoffs_p2_game1 = payoff_p2_game_1flatten.size
    total_payoffs_p2_game2 = payoff_p2_game_2flatten.size
    total_payoffs_p2 = total_payoffs_p2_game1 + total_payoffs_p2_game2

    # initialize the payoffs for p2 and assign them
    payoff_p2 = np.zeros(total_payoffs_p2)
    payoff_p2[0:total_payoffs_p2_game1] = payoff_p2_game_1flatten
    payoff_p2[total_payoffs_p2_game1:total_payoffs_p2] = payoff_p2_game_2flatten

    px = np.concatenate([flatten1_1, flatten2_1], axis=1)  # trix with px

    x_punisher = random_strategy_draw(points, total_actions_p1)  # draw some awesome strategies

    frequency_pairs = frequency_pairs_p2(points, total_actions_p2, total_actions_p1,
                                         x_punisher)  # sort them based on best replies

    # do some balance equation accelerator magic
    frequency_pairs = balance_equation(self, actions_p2_game1, actions_p2_game2, total_payoffs_p2_game1,
                                       total_payoffs_p2, frequency_pairs)

    # activate FD function
    if self.FD == True:
        if self.FD_function_use == "FD":
            FD = self.FD_function(draw_payoffs)
        elif self.FD_function_use == "mu":
            FD = self.mu_function(self.rho_function(draw_payoffs))
    else:
        FD = 1

        # payoffs are calculated
    payoffs = np.sum(np.multiply(frequency_pairs, payoff_p2), axis=1)

    if self.rarity == True:
        rho = self.rho_function(frequency_pairs)
        mu = self.mu_function(rho)
        payoffs = np.multiply(self.profit_function(mu), payoffs)
    else:
        # compute the payoffs with payoffs and FD function
        payoffs = np.multiply(FD, payoffs)
        payoffs = payoffs.reshape((payoffs.size, 1))

    max_payoffs = payoffs_sorted(points, payoffs, (actions_p2_game1 * actions_p2_game2))  # awesome sorting process
    nan_delete = np.where(np.isnan(max_payoffs))  # look for NaN's

    max_payoffs_p2 = np.delete(max_payoffs, nan_delete[0], 0)  # delete them where necessary
    threat_point_p2 = np.nanmin(np.nanmax(max_payoffs_p2, axis=1))  # determine the threat point

    location_threat_p2 = np.where(threat_point_p2 == max_payoffs_p2)

    #         if self.rarity == True:
    #             select_payoffs_p2 = np.multiply(self.profit_function(FD),select_payoffs_p2)
    #             select_payoffs_p2 = select_payoffs_p2.reshape((select_payoffs_p2.size,1))

    #             second_sort_p2 = payoffs_sorted(points,select_payoffs_p2,(actions_p2_game1*actions_p2_game2))

    #             second_sort_p2 = np.delete(second_sort_p2,nan_delete[0],0)
    #             location_x = location_threat_p2[0]
    #             location_y = location_threat_p2[1]
    #             threat_point_p2 = (second_sort_p2[location_x,location_y])[0]

    if print_text == True:
        print("")
        print("")
        print("Threat point value is", threat_point_p2)
        print("")
        print("")

    if show_strat_p2 == True:
        threat_point_indices_p2 = np.where(max_payoffs_p2 == threat_point_p2)
        found_strategy = x_punisher[threat_point_indices_p2[0]]
        fnd_strategy = found_strategy.flatten()
        fnd_strategy[0:2] = fnd_strategy[0:2] / np.sum(fnd_strategy[0:2])
        fnd_strategy[2:4] = fnd_strategy[2:4] / np.sum(fnd_strategy[2:4])
        print("Player 1 plays stationairy strategy:", fnd_strategy)
        print("While player 2 replies with a best pure reply of:",
              self.best_pure_strategies[threat_point_indices_p2[1]])

    end_time_p2 = time.time()  # stop the time

    if print_text == True:
        print("")
        print("Seconds done to generate", points, "points", end_time_p2 - start_time_p2)
        print("")
        print("")

    self.threat_point = np.zeros(2)
    self.threat_point = [threat_point_p1, threat_point_p2]  # store the threat point!

    return [threat_point_p1, threat_point_p2]