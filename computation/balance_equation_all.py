import numpy as np

from accelerators.aitken_delta_squared import aitken_delta_squared


def balance_equation_all(self, points, tot_pay_p1_g1, tot_pay_p1_tot, draw_payoffs):
    yi = np.zeros((points, self.payoff_p1_actions))
    Q = np.zeros((1, points))
    Q_new = np.zeros((1, points))

    yi[:, 0:tot_pay_p1_g1] = draw_payoffs[:, 0:tot_pay_p1_g1] / np.sum(
        draw_payoffs[:, 0:tot_pay_p1_g1], axis=1).reshape([points, 1])
    yi[:, tot_pay_p1_g1:tot_pay_p1_tot] = draw_payoffs[:,
                                                     tot_pay_p1_g1:tot_pay_p1_tot] / np.sum(
        draw_payoffs[:, tot_pay_p1_g1:tot_pay_p1_tot], axis=1).reshape([points, 1])

    index_values = np.arange(points)

    p1_px_between = np.asarray(self.px)
    p1_px = p1_px_between[0]

    if self.hysteresis:
        etp_calculation = np.multiply(self.phi, self.etp_matrix)
    else:
        etp_calculation = self.etp_matrix

    for i in np.arange(35):

        if i == 0:
            new_x = p1_px - np.dot(draw_payoffs, etp_calculation)

            upper_part_Q = np.sum(np.multiply(yi[:, tot_pay_p1_g1:tot_pay_p1_tot],
                                              new_x[:, tot_pay_p1_g1:tot_pay_p1_tot]), axis=1)
            leftdown_part_Q = np.sum(
                np.multiply(yi[:, 0:tot_pay_p1_g1], (1 - new_x[:, 0:tot_pay_p1_g1])), axis=1)
            rightdown_part_Q = np.sum(np.multiply(yi[:, tot_pay_p1_g1:tot_pay_p1_tot],
                                                  new_x[:, tot_pay_p1_g1:tot_pay_p1_tot]), axis=1)

            Q_between = upper_part_Q / (leftdown_part_Q + rightdown_part_Q)
            Q = Q_between

            draw_payoffs[:, 0:tot_pay_p1_g1] = (np.multiply(Q, yi[:, 0:tot_pay_p1_g1]))
            draw_payoffs[:, tot_pay_p1_g1:tot_pay_p1_tot] = np.multiply((1 - Q), yi[:,
                                                                                            tot_pay_p1_g1:tot_pay_p1_tot])

        if i > 0 and i < 10:
            new_x = p1_px - np.dot(draw_payoffs, etp_calculation)

            upper_part_Q = np.sum(np.multiply(yi[:, tot_pay_p1_g1:tot_pay_p1_tot],
                                              new_x[:, tot_pay_p1_g1:tot_pay_p1_tot]), axis=1)
            leftdown_part_Q = np.sum(
                np.multiply(yi[:, 0:tot_pay_p1_g1], (1 - new_x[:, 0:tot_pay_p1_g1])), axis=1)
            rightdown_part_Q = np.sum(np.multiply(yi[:, tot_pay_p1_g1:tot_pay_p1_tot],
                                                  new_x[:, tot_pay_p1_g1:tot_pay_p1_tot]), axis=1)

            Q_between = upper_part_Q / (leftdown_part_Q + rightdown_part_Q)
            Q = np.hstack((Q, Q_between))

            draw_payoffs[:, 0:tot_pay_p1_g1] = (np.multiply(Q[:, i], yi[:, 0:tot_pay_p1_g1]))
            draw_payoffs[:, tot_pay_p1_g1:tot_pay_p1_tot] = np.multiply((1 - Q[:, i]), yi[:,
                                                                                                  tot_pay_p1_g1:tot_pay_p1_tot])

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

    draw_payoffs[:, 0:tot_pay_p1_g1] = (np.multiply(Q[:, 34], yi[:, 0:tot_pay_p1_g1]))
    draw_payoffs[:, tot_pay_p1_g1:tot_pay_p1_tot] = np.multiply((1 - Q[:, 34]), yi[:,
                                                                                           tot_pay_p1_g1:tot_pay_p1_tot])
    return draw_payoffs