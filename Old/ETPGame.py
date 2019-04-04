import numpy as np  #import numpy
from scipy.spatial import ConvexHull #import scipy convex hull package
import matplotlib.pyplot as plt #import package to plot stuff
import time #import time package


class ETPGame:
    "The ETP Game class represents the Type III games from the thesis, with or without ESP."

    def __init__(self, payoff_p1_game1, payoff_p2_game1, payoff_p1_game2, payoff_p2_game2, trmatrixg1, trmatrixg2,
                 trmatrixg3, trmatrixg4, matrixA):
        "Here below we initialize the game by storing payoff and transition matrices according to the upper input."
        self.payoff_p1_game1 = payoff_p1_game1  # payoff p1 in game 1
        self.payoff_p2_game1 = payoff_p2_game1  # payoff p2 in game 1

        self.payoff_p1_game2 = payoff_p1_game2  # payoff p1 in game 2
        self.payoff_p2_game2 = payoff_p2_game2  # payoff p2 in game 2

        self.transition_matrix_game1_to1 = trmatrixg1  # transition matrix from game 1 to game 1
        self.transition_matrix_game2_to1 = trmatrixg2  # transition matrix from game 2 to game 1

        self.transition_matrix_game1_to2 = trmatrixg3  # transition matrix from game 1 to game 2
        self.transition_matrix_game2_to2 = trmatrixg4  # transition matrix from game 2 to game 2

        self.etp_matrix = matrixA

        self.printing = False  # set printing to False

        self.best_pure_strategies = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]])

    def activate_FD(self):

        print("Game is now an ESP game")

        self.FD = True
        self.FD_function_use = "FD"

    def deactivate_FD(self):

        print("FD function deactivated")
        self.FD = False

    def activate_rarity(self):

        print("Rarity function active")

        self.rarity = True

    def deactivate_rarity(self):

        print("Rarity function deactivated")

        self.rarity = False

    def activate_hysteresis(self, phi):

        print("Hysteresis is now active")

        self.hysteresis = True
        self.phi = phi

    def deactivate_hysteresis(self):

        print("Hysteresis disabled")

        self.hysteresis = False

    def adjust_mu(self, m):

        self.m = m

        print("Mu adjusted, now has:")
        print("M = ", m)

    def adjust_FD(self, type_function):

        if type_function == "mu":
            self.FD_function_use = "mu"
            self.m = 0
        else:
            self.FD_function_use = "FD"

    def FD_function(self, x):

        FD = 1 - 0.25 * (x[:, 1] + x[:, 2]) - (1 / 3) * x[:, 3] - (1 / 2) * (x[:, 5] + x[:, 6]) - (2 / 3) * x[:, 7]

        return FD

    def plotting_rarity(self, plot):

        if plot == "Rarity":
            self.plotting_rarity = plot
            self.m = 1
        elif plot == "Revenue":
            self.plotting_rarity = plot
            self.m = 1
        else:
            self.plotting_rarity = False

        print("Plotting rarity is now:", self.plotting_rarity)

    def rho_function(self, x):

        rho = float(3 / 8) * (x[:, 1] + x[:, 2] + 2 * (x[:, 5] + x[:, 6])) + float(1 / 2) * (x[:, 3] + 2 * x[:, 7])

        return rho

    def mu_function(self, rho):
        m = self.m
        #         mu = 1 + (1 - m) * ((n2/(n1-n2))*rho**n1 - (n1/(n1-n2))*rho**n2)
        mu = 1 + (1 - m) * (2 * rho ** 3 - 3 * rho ** 2)

        return mu

    def profit_function(self, mu):

        first_part = np.divide(1, 3.75)
        second_part = 4 + np.multiply(0.75, np.divide(1, np.power(mu, 2)))
        third_part = 12 + np.multiply((1 - 12 * mu ** 2.5), np.divide(1, np.power(mu, 1.5)))

        combined_last = second_part - third_part

        return np.multiply(first_part, combined_last)

    def plot_single_period_pure_rewards(self):
        "Here we plot the pure rewards possible for a single period"

        plt.figure()  # create a figure
        payoff_p1_g1_flat = self.payoff_p1_game1.A1  # create a flattend payoff of p1 in game 1
        payoff_p2_g1_flat = self.payoff_p2_game1.A1  # create a flattend payoff of p2 in game 1
        plt.scatter(payoff_p1_g1_flat, payoff_p2_g1_flat, label="Pure reward points Game 1",
                    zorder=15)  # plot payoffs game 1

        payoff_p1_g2_flat = self.payoff_p1_game2.A1  # create a flattend payoff of p1 in game 2
        payoff_p2_g2_flat = self.payoff_p2_game2.A1  # and for p2 in game 2
        plt.scatter(payoff_p1_g2_flat, payoff_p2_g2_flat, label="Pure reward points Game 2",
                    zorder=15)  # plotting this again

        plt.xlabel("Payoff Player 1")  # giving the x-axis the label of payoff p1
        plt.ylabel("Payoff Player 2")  # and the payoff of the y-axis is that of p2
        plt.title("Reward points of ETP game")  # and we give it a nice titel
        plt.legend()

    def plot_convex_hull_pure_rewards(self):
        "Here we plot a convex hull around the pure reward point, therefore resulting in the total possible reward space"

        payoff_p1_g1_flat = self.payoff_p1_game1.A1  # store the flattend payoff of p1 game 1
        payoff_p2_g1_flat = self.payoff_p2_game1.A1  # store the flattend payoff of p2 game 1

        payoff_p1_g2_flat = self.payoff_p1_game2.A1  # store the flattend payoff of p1 game 2
        payoff_p2_g2_flat = self.payoff_p2_game2.A1  # store the flattend payoff of p2 game 2

        payoff_p1_merged = np.concatenate((payoff_p1_g1_flat, payoff_p1_g2_flat))  # merge p1 payoffs
        payoff_p2_merged = np.concatenate((payoff_p2_g1_flat, payoff_p2_g2_flat))  # merge p2 payoffs

        all_payoffs = np.array([payoff_p1_merged, payoff_p2_merged])  # create one array of payoffs
        all_payoffs = np.transpose(all_payoffs)  # and rotate this one

        rewards_convex_hull = ConvexHull(all_payoffs)  # retain the convex hull of the payoffs
        plt.fill(all_payoffs[rewards_convex_hull.vertices, 0], all_payoffs[rewards_convex_hull.vertices, 1], color='k')
        # here above we fill the convex hull in black

    def plot_all_rewards(self, points):

        print("Now plotting all rewards")

        start_time = time.time()

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

        start_time = time.time()  # timer start

        # flatten payoffs game 1 and 2
        payoff_p1_game_1flatten = self.payoff_p1_game1.flatten()
        payoff_p1_game_2flatten = self.payoff_p1_game2.flatten()

        payoff_p2_game_1flatten = self.payoff_p2_game1.flatten()
        payoff_p2_game_2flatten = self.payoff_p2_game2.flatten()

        # store size of the payoffs
        total_payoffs_p1_game1 = payoff_p1_game_1flatten.size
        total_payoffs_p1_game2 = payoff_p1_game_2flatten.size
        total_payoffs_p1 = total_payoffs_p1_game1 + total_payoffs_p1_game2

        # initialize and assign payoffs
        payoff_p1 = np.zeros(total_payoffs_p1)
        payoff_p1[0:total_payoffs_p1_game1] = payoff_p1_game_1flatten
        payoff_p1[total_payoffs_p1_game1:total_payoffs_p1] = payoff_p1_game_2flatten

        payoff_p2 = np.zeros(total_payoffs_p1)
        payoff_p2[0:total_payoffs_p1_game1] = payoff_p2_game_1flatten
        payoff_p2[total_payoffs_p1_game1:total_payoffs_p1] = payoff_p2_game_2flatten

        px = np.concatenate([flatten1_1, flatten2_1], axis=1)  # store px

        ## Build a draw function

        draw_payoffs = np.zeros((points, total_payoffs_p1))
        draw_payoffs = np.random.beta(0.5, 0.5, (points, total_payoffs_p1))
        draw_payoffs = draw_payoffs / np.sum(draw_payoffs, axis=1).reshape([points, 1])

        ## Calculate the balance equations

        yi = np.zeros((points, total_payoffs_p1))
        Q = np.zeros((1, points))
        Q_new = np.zeros((1, points))

        yi[:, 0:total_payoffs_p1_game1] = draw_payoffs[:, 0:total_payoffs_p1_game1] / np.sum(
            draw_payoffs[:, 0:total_payoffs_p1_game1], axis=1).reshape([points, 1])
        yi[:, total_payoffs_p1_game1:total_payoffs_p1] = draw_payoffs[:,
                                                         total_payoffs_p1_game1:total_payoffs_p1] / np.sum(
            draw_payoffs[:, total_payoffs_p1_game1:total_payoffs_p1], axis=1).reshape([points, 1])

        index_values = np.arange(points)

        p1_px_between = np.asarray(px)
        p1_px = p1_px_between[0]

        if self.hysteresis == True:
            etp_calculation = np.multiply(self.phi, self.etp_matrix)
        else:
            etp_calculation = self.etp_matrix

        for i in np.arange(35):

            if i == 0:
                new_x = p1_px - np.dot(draw_payoffs, etp_calculation)

                upper_part_Q = np.sum(np.multiply(yi[:, total_payoffs_p1_game1:total_payoffs_p1],
                                                  new_x[:, total_payoffs_p1_game1:total_payoffs_p1]), axis=1)
                leftdown_part_Q = np.sum(
                    np.multiply(yi[:, 0:total_payoffs_p1_game1], (1 - new_x[:, 0:total_payoffs_p1_game1])), axis=1)
                rightdown_part_Q = np.sum(np.multiply(yi[:, total_payoffs_p1_game1:total_payoffs_p1],
                                                      new_x[:, total_payoffs_p1_game1:total_payoffs_p1]), axis=1)

                Q_between = upper_part_Q / (leftdown_part_Q + rightdown_part_Q)
                Q = Q_between

                draw_payoffs[:, 0:total_payoffs_p1_game1] = (np.multiply(Q, yi[:, 0:total_payoffs_p1_game1]))
                draw_payoffs[:, total_payoffs_p1_game1:total_payoffs_p1] = np.multiply((1 - Q), yi[:,
                                                                                                total_payoffs_p1_game1:total_payoffs_p1])

            if i > 0 and i < 10:
                new_x = p1_px - np.dot(draw_payoffs, etp_calculation)

                upper_part_Q = np.sum(np.multiply(yi[:, total_payoffs_p1_game1:total_payoffs_p1],
                                                  new_x[:, total_payoffs_p1_game1:total_payoffs_p1]), axis=1)
                leftdown_part_Q = np.sum(
                    np.multiply(yi[:, 0:total_payoffs_p1_game1], (1 - new_x[:, 0:total_payoffs_p1_game1])), axis=1)
                rightdown_part_Q = np.sum(np.multiply(yi[:, total_payoffs_p1_game1:total_payoffs_p1],
                                                      new_x[:, total_payoffs_p1_game1:total_payoffs_p1]), axis=1)

                Q_between = upper_part_Q / (leftdown_part_Q + rightdown_part_Q)
                Q = np.hstack((Q, Q_between))

                draw_payoffs[:, 0:total_payoffs_p1_game1] = (np.multiply(Q[:, i], yi[:, 0:total_payoffs_p1_game1]))
                draw_payoffs[:, total_payoffs_p1_game1:total_payoffs_p1] = np.multiply((1 - Q[:, i]), yi[:,
                                                                                                      total_payoffs_p1_game1:total_payoffs_p1])

            if i == 10:
                Q_new = self.aitken_delta_squared(Q[:, i - 3], Q[:, i - 2], Q[:, i - 1])
                nan_org = np.where(np.isnan(Q_new))
                nan_indic = nan_org[0]
                Q_new[nan_indic, :] = Q_between[nan_indic, :]

                Q_old = np.copy(Q_new)
                Q = np.hstack((Q, Q_new))

            if i > 10:
                Q_new[index_values, :] = self.aitken_delta_squared(Q[index_values, i - 3], Q[index_values, i - 2],
                                                                   Q[index_values, i - 1])
                Q_old2 = np.copy(Q_old)
                nan_res = np.where(np.isnan(Q_new, Q_old))
                nan_indices = nan_res[0]

                nan_between = np.where(np.in1d(index_values, nan_indices))
                nan_cands = nan_between[0]
                index_values = np.delete(index_values, nan_cands)

                Q_new[nan_indices, :] = Q_old2[nan_indices, :]

                Q = np.hstack((Q, Q_new))

                Q_old = np.copy(Q_new)
                results = np.where(Q[index_values, i - 1] == Q[index_values, i])
                remove_indices = results[0]

                removal_between = np.where(np.in1d(index_values, remove_indices))
                removal_cands = removal_between[0]
                index_values = np.delete(index_values, removal_cands)

        draw_payoffs[:, 0:total_payoffs_p1_game1] = (np.multiply(Q[:, 34], yi[:, 0:total_payoffs_p1_game1]))
        draw_payoffs[:, total_payoffs_p1_game1:total_payoffs_p1] = np.multiply((1 - Q[:, 34]), yi[:,
                                                                                               total_payoffs_p1_game1:total_payoffs_p1])

        # activate the FD function
        if self.FD == True:
            if self.FD_function_use == "FD":
                FD = self.FD_function(draw_payoffs)
            elif self.FD_function_use == "mu":
                FD = self.mu_function(self.rho_function(draw_payoffs))
        else:
            FD = 1

        payoffs_p1 = np.sum(np.multiply(draw_payoffs, payoff_p1), axis=1)
        payoffs_p2 = np.sum(np.multiply(draw_payoffs, payoff_p2), axis=1)

        #         payoffs_p1 = np.multiply(self.profit_function(FD),payoffs_p1)
        #         payoffs_p2 = np.multiply(self.profit_function(FD),payoffs_p2)

        if self.plotting_rarity == "Rarity":
            print("Plotting with rarity active")
            rho = self.rho_function(draw_payoffs)
            mu = self.mu_function(rho)
            payoffs_p1 = np.multiply(self.profit_function(mu), payoffs_p1)
            payoffs_p2 = np.multiply(self.profit_function(mu), payoffs_p2)
        elif self.plotting_rarity == "Revenue":
            print("Plotting with revenue active")
            payoffs_p1 = np.multiply(self.revenue_function(FD), payoffs_p1)
            payoffs_p2 = np.multiply(self.revenue_function(FD), payoffs_p2)
        else:
            print("Normal plotting active")
            payoffs_p1 = np.multiply(FD, payoffs_p1)
            payoffs_p2 = np.multiply(FD, payoffs_p2)

        delete_indic = np.where(np.isnan(payoffs_p1))
        payoffs_p1 = np.delete(payoffs_p1, delete_indic[0], 0)
        payoffs_p2 = np.delete(payoffs_p2, delete_indic[0], 0)

        self.maximal_payoffs = np.zeros(2)
        self.maximal_payoffs = [np.max(payoffs_p1), np.max(payoffs_p2)]

        self.minimal_payoffs = np.zeros(2)
        self.minimal_payoffs = [np.min(payoffs_p1), np.min(payoffs_p2)]

        all_payoffs = np.array([payoffs_p1, payoffs_p2])
        all_payoffs = np.transpose(all_payoffs)
        Convex_Hull_Payoffs = ConvexHull(all_payoffs, qhull_options='QbB')

        plt.scatter(payoffs_p1, payoffs_p2)
        #         plt.fill(all_payoffs[Convex_Hull_Payoffs.vertices,0],all_payoffs[Convex_Hull_Payoffs.vertices,1],color='y', zorder=5, label="Obtainable rewards")

        end_time = time.time()

        print("Total time taken to plot all reward points:", end_time - start_time)

    def plot_threat_point(self):
        "This function plots the threat point if found"
        plt.scatter(self.threat_point[0], self.threat_point[1], zorder=10, color='r', label='Threat point')
        plt.legend()

    def plot_threat_point_lines(self):
        "This function plots lines around the threat point indicating the limits for the NE"

        plt.plot([self.threat_point[0], self.threat_point[0]], [self.threat_point[1], self.maximal_payoffs[1]],
                 color='k', zorder=15)
        plt.plot([self.threat_point[0], self.maximal_payoffs[0]], [self.threat_point[1], self.threat_point[1]],
                 color='k', zorder=15)

    def optimized_maximin(self, points, show_strat_p1, show_strat_p2, FD_yn):
        "This is an optimized version for determining the maximin result"

        print("Start of the maximin algorithm")

        def random_strategy_draw(points, number_of_actions):
            "This function draws random strategies from a beta distribution, based on the number of points and actions"

            # draw some strategies and normalize them
            strategies_drawn = np.random.beta(0.5, 0.5, (points, number_of_actions))
            strategies_drawn = strategies_drawn / np.sum(strategies_drawn, axis=1).reshape([points, 1])

            return strategies_drawn

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

                    # delete values which are NaN for future computations
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

"ETP Example game as described in the thesis, based on a game developed by Llea Samuel"
p1_1 = np.matrix('16 14; 28 24')
p2_1 = np.matrix('16 28; 14 24')

p1_2 = np.matrix('4 3.5; 7 6')
p2_2 = np.matrix('4 7; 3.5 6')

trans1_1 = np.matrix('0.8 0.7; 0.7 0.6')
trans2_1 = np.matrix('0.5 0.4; 0.4 0.15')

trans1_2 = np.matrix('0.2 0.3; 0.3 0.4')
trans2_2 = np.matrix('0.5 0.6; 0.6 0.85')

matrixA = np.matrix('0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.35 0.3 0.3 0.25 0.2 0.15 0.15 0.05; 0.35 0.3 0.3 0.25 0.2 0.15 0.15 0.05; 0.7 0.6 0.6 0.5 0.4 0.3 0.3 0.1; 0 0 0 0 0 0 0 0; 0.35 0.3 0.3 0.25 0.2 0.15 0.15 0.05; 0.35 0.3 0.3 0.25 0.2 0.15 0.15 0.05; 0.7 0.6 0.6 0.5 0.4 0.3 0.3 0.1')
matrixB = np.matrix('0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.0 0.0 0.0 0.00 0.0 0.0 0.0 0.00; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0 0 0 0 0 0 0 0; 0.0 0.0 0.0 0.00 0.0 0.00 0.00 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.00; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0')
matrixC = np.matrix('0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.57 0.5 0.5 0.45 0.39 0.315 0.315 0.12; 0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.57 0.5 0.5 0.45 0.39 0.315 0.315 0.12')

phi_range = np.arange(0,0.05,0.01)
# mu_range = np.arange(0,2,1)
# found_maximum = 0
# found_minimum = 0

FirstTryETP = ETPGame(p1_1,p2_1,p1_2,p2_2,trans1_1,trans2_1,trans1_2,trans2_2,matrixC)
FirstTryETP.activate_FD()
FirstTryETP.adjust_FD("mu")
FirstTryETP.adjust_mu(1)
k = 0
# FirstTryETP.activate_rarity()
# FirstTryETP.plotting_rarity("Rarity")

for i in np.arange(phi_range.size):
    plt.close()
    print("Now plotting with phi =", phi_range[i])
    FirstTryETP.activate_hysteresis(phi_range[i])
#     print("And with m = ", mu_range[j])
#     FirstTryETP.adjust_mu(mu_range[j])
#     FirstTryETP.threat_point_optimized(100000,True,True,True,True)
    # FirstTryETP.plot_single_period_pure_rewards()
    FirstTryETP.plot_all_rewards(500000)
#     FirstTryETP.plot_threat_point()
#     FirstTryETP.plot_threat_point_lines()
    title = "Small Fish Wars with Hysteresis, 500k points"
    plt.title(title)

#     plt.axis([-15, 430, -15, 430])
#     plt.figtext(0,0,'Threat point: ' + str(FirstTryETP.threat_point))
    plt.figtext(0,-0.05,'Minimal rewards: ' + str(FirstTryETP.minimal_payoffs))
    plt.figtext(0,-0.1,'Maximal rewards: ' + str(FirstTryETP.maximal_payoffs))
    plt.figtext(0,-0.15,'With hysteresis phi at: ' + str(FirstTryETP.phi))
    plt.figtext(0,-0.2,'And m at: ' + str(FirstTryETP.m))

    plt.savefig('figures/without_convex_%d.png'%(k), dpi=300, bbox_inches = "tight")
    k += 1