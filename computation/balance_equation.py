import numpy as np

from accelerators.aitken_delta_squared import aitken_delta_squared

def balance_equation(self, points, tot_act_ut, tot_act_thr, tot_payoffs_game1, tot_payoffs, frequency_pairs):
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

    p1_px_between = np.asarray(self.px)  # set px
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
            Q_new = aitken_delta_squared(Q[:, i - 3], Q[:, i - 2], Q[:, i - 1])
            nan_org = np.where(np.isnan(Q_new))  # check whether Nan's occur
            nan_indic = nan_org[0]

            Q_new[nan_indic, :] = Q_between[nan_indic, :]  # replace NaN with last known value

            Q_old = np.copy(Q_new)

            Q = np.hstack((Q, Q_new))

        # and only Aitken's
        if i > 10:
            Q_new[index_values, :] = aitken_delta_squared(Q[index_values, i - 3], Q[index_values, i - 2],
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