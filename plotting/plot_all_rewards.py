import time
import numpy as np
import matplotlib.pyplot as plt

from accelerators.aitken_delta_squared import aitken_delta_squared
from FD_functions.fd_function import fd_function

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
    yi[:, total_payoffs_p1_game1:total_payoffs_p1] = draw_payoffs[:, total_payoffs_p1_game1:total_payoffs_p1] / np.sum(
        draw_payoffs[:, total_payoffs_p1_game1:total_payoffs_p1], axis=1).reshape([points, 1])

    index_values = np.arange(points)

    p1_px_between = np.asarray(px)
    p1_px = p1_px_between[0]

    if self.hysteresis:
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
            Q_new = aitken_delta_squared(Q[:, i - 3], Q[:, i - 2], Q[:, i - 1])
            nan_org = np.where(np.isnan(Q_new))
            nan_indic = nan_org[0]
            Q_new[nan_indic, :] = Q_between[nan_indic, :]

            Q_old = np.copy(Q_new)
            Q = np.hstack((Q, Q_new))

        if i > 10:
            Q_new[index_values, :] = aitken_delta_squared(Q[index_values, i - 3], Q[index_values, i - 2],
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
    if self.FD:
        if self.FD_function_use == "FD":
            FD = fd_function(draw_payoffs)
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
    # Convex_Hull_Payoffs = ConvexHull(all_payoffs, qhull_options='QbB')

    plt.scatter(payoffs_p1, payoffs_p2, s=0.3)
    #         plt.fill(all_payoffs[Convex_Hull_Payoffs.vertices,0],all_payoffs[Convex_Hull_Payoffs.vertices,1],color='y', zorder=5, label="Obtainable rewards")
    end_time = time.time()
    plt.show()

    print("Total time taken to plot all reward points:", end_time - start_time)
