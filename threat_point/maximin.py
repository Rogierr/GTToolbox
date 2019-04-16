def optimized_maximin(self, points, show_strat_p1, show_strat_p2, FD_yn):
    "This is an optimized version for determining the maximin result"

    print("Start of the maximin algorithm")

    def frequency_pairs_p1(points, p2_actions, p1_actions, strategies_drawn):
        "Create strategies based on the best replies for player 1"

        # store the size of the games
        game_size_1 = self.payoff_p1_game1.size
        game_size_2 = self.payoff_p1_game2.size

        # store the actions for game 1 and 2
        p1_actions_game1 = self.payoff_p1_game1.shape[0]
        p1_actions_game2 = self.payoff_p1_game2.shape[0]

        # calculate the combination of the actions and a range
        p1_actions_combi = p1_actions_game1 * p1_actions_game2
        p1_action_range = np.arange(p1_actions_combi)

        # initialize a frequency pair
        frequency_pairs = np.zeros((points * (p1_actions_game1 * p1_actions_game2), game_size_1 + game_size_2))

        # create actions ranges
        p1_act_game1_range = np.arange(p1_actions_game1)
        p1_act_game2_range = np.arange(p1_actions_game2)

        # loop over best responses for game 1
        for i in np.nditer(p1_action_range):
            for j in np.nditer(p1_act_game1_range):
                mod_remain = np.mod(i, p1_actions_game1)
                frequency_pairs[i * points:(i + 1) * points, p1_actions_game1 * mod_remain + j] = strategies_drawn[
                                                                                                  :, j]

        # loop over best responses for game 2
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

        # compute yi
        yi[:, 0:tot_payoffs_game1] = frequency_pairs[:, 0:tot_payoffs_game1] / np.sum(
            frequency_pairs[:, 0:tot_payoffs_game1], axis=1).reshape([points * tot_payoffs_game1, 1])
        yi[:, tot_payoffs_game1:tot_payoffs] = frequency_pairs[:, tot_payoffs_game1:tot_payoffs] / np.sum(
            frequency_pairs[:, tot_payoffs_game1:tot_payoffs], axis=1).reshape(
            [points * (tot_payoffs - tot_payoffs_game1), 1])

        index_values = np.arange(points * (tot_act_thr * tot_act_ut))  # set a range

        # set px range
        p1_px_between = np.asarray(px)
        p1_px = p1_px_between[0]

        # loop for 35 iterations
        for i in np.arange(35):

            # in the first iteration we calculate the first Q and adjust X
            if i == 0:
                new_x = p1_px - np.dot(frequency_pairs, self.etp_matrix)

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

            # here we just calculate Q in order to guarantee stability
            if i > 0 and i < 10:
                new_x = p1_px - np.dot(frequency_pairs, self.etp_matrix)

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

            # here we calculate Q based on aitken's delta squared
            if i == 10:
                Q_new = self.aitken_delta_squared(Q[:, i - 3], Q[:, i - 2], Q[:, i - 1])
                nan_org = np.where(np.isnan(Q_new))  # check where there are NaN's
                nan_indic = nan_org[0]
                Q_new[nan_indic, :] = Q_between[nan_indic, :]  # remove NaN's with last known values

                Q_old = np.copy(Q_new)

                Q = np.hstack((Q, Q_new))

            # all remaining iterations are with Aitkens
            if i > 10:
                Q_new[index_values, :] = self.aitken_delta_squared(Q[index_values, i - 3], Q[index_values, i - 2],
                                                                   Q[index_values, i - 1])
                Q_old2 = np.copy(Q_old)
                nan_res = np.where(np.isnan(Q_new, Q_old))  # check for NaN's
                nan_indices = nan_res[0]  # look where NaN's are

                # delete values which are NaN for future computation
                nan_between = np.where(np.in1d(index_values, nan_indices))
                nan_cands = nan_between[0]
                index_values = np.delete(index_values, nan_cands)

                Q_new[nan_indices, :] = Q_old2[nan_indices, :]

                Q = np.hstack((Q, Q_new))

                Q_old = np.copy(Q_new)
                results = np.where(
                    Q[index_values, i - 1] == Q[index_values, i])  # check where convergence has occured
                remove_indices = results[0]

                # remove indices which have converged
                removal_between = np.where(np.in1d(index_values, remove_indices))
                removal_cands = removal_between[0]
                index_values = np.delete(index_values, removal_cands)

        # compute the definitive frequency pair x
        frequency_pairs[:, 0:tot_payoffs_game1] = (np.multiply(Q[:, 34], yi[:, 0:tot_payoffs_game1]))
        frequency_pairs[:, tot_payoffs_game1:tot_payoffs] = np.multiply((1 - Q[:, 34]),
                                                                        yi[:, tot_payoffs_game1:tot_payoffs])

        return frequency_pairs

    def frequency_pairs_p2(points, p2_actions, p1_actions, strategies_drawn):
        "Best responses for P2 based on threaten strategies drawn"

        # store the game sizes
        game_size_1 = self.payoff_p2_game1.size
        game_size_2 = self.payoff_p2_game2.size

        # store the actions for p1 and p2 and create ranges
        p1_actions_range = np.arange(p1_actions)
        p2_actions_range = np.arange(p2_actions)

        p2_actions_game1 = self.payoff_p2_game1.shape[1]
        p2_actions_game2 = self.payoff_p2_game2.shape[1]

        p2_actions_combo = p2_actions_game1 * p2_actions_game2
        p2_action_range = np.arange(p2_actions_combo)

        # initialize the frequency pair
        frequency_pairs = np.zeros((points * (p2_actions_game1 * p2_actions_game2), game_size_1 + game_size_2))

        # generate best responses for game 1
        for i in np.nditer(np.arange(p2_actions_game1)):
            for j in np.nditer(p2_action_range):
                modul = np.mod(j, p2_actions_game1)
                frequency_pairs[j * points:(j + 1) * points, p2_actions_game1 * i + modul] = strategies_drawn[:, i]

        # generate best respones for game 2
        for i in np.nditer(np.arange(p2_actions_game2)):
            for j in np.nditer(p2_action_range):
                divide = np.floor_divide(j, p2_actions_game2)
                frequency_pairs[j * points:(j + 1) * points,
                p2_actions_combo + divide + (i * p2_actions_game2)] = strategies_drawn[:, i + p2_actions_game1]

        return frequency_pairs

    def payoffs_sorted(points, payoffs, actions):
        "Sort the payoffs for determination of maximin"

        # store the range of points and actions
        points_range = np.arange(points)
        actions_range = np.arange(actions)

        payoffs_sort = np.zeros((points, actions))  # initialize the payoffs sort

        # sort the payoffs!
        for x in np.nditer(points_range):
            for i in np.nditer(actions_range):
                payoffs_sort[x, i] = payoffs[points * i + x]

        return payoffs_sort

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

    frequency_pairs = frequency_pairs_p2(points, total_actions_p1, total_actions_p2,
                                         y_punisher)  # sort them based on best replies

    # do the balance equations with Aitken's
    frequency_pairs = balance_equation(self, actions_p2_game1, actions_p2_game2, total_payoffs_p2_game1,
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

    frequency_pairs = frequency_pairs_p1(points, total_actions_p1, total_actions_p2, x_punisher)  # best reponses p1

    # balance equations with Delta Squared
    frequency_pairs = balance_equation(self, actions_p1_game1, actions_p1_game2, total_payoffs_p1_game1,
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