import numpy as np
from accelerators.aitken_delta_squared import aitken_delta_squared

x = np.array([0, 0, 0, 1, 0, 0, 0, 0,])
pi = np.array([0.8, 0.7, 0.7, 0.6, 0.5, 0.4, 0.4, 0.15])
matrixC = np.matrix('0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06;'
                    ' 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.57 0.5 0.5 0.45 0.39 0.315 0.315 0.12;'
                    ' 0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06;'
                    ' 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.57 0.5 0.5 0.45 0.39 0.315 0.315 0.12')

draw_payoffs = np.array([0., 1., 0., 0., 0., 1., 0., 0.])

yi = np.zeros(8)
Q = np.zeros(1)
Q_new = np.zeros(1)

yi[0:4] = draw_payoffs[0:4] / np.sum(
    draw_payoffs[0:4])
yi[4:8] = draw_payoffs[4:8] / np.sum(
    draw_payoffs[4:8])

index_values = np.arange(8)

p1_px_between = np.asarray(pi)
p1_px = p1_px_between[0]

etp_calculation = matrixC

for i in np.arange(35):

    if i == 0:
        new_x = p1_px - np.dot(draw_payoffs, etp_calculation)
        np.place(new_x, new_x < 0, 0)
        np.place(new_x, new_x > 1, 1)

        upper_part_Q = np.sum(np.multiply(yi[4:8], new_x[:, 4:8]))
        leftdown_part_Q = np.sum(
            np.multiply(yi[0:4], (1 - new_x[:, 0:4])))
        rightdown_part_Q = np.sum(np.multiply(yi[4:8],
                                              new_x[:, 4:8]))

        Q_between = upper_part_Q / (leftdown_part_Q + rightdown_part_Q)
        Q = Q_between

        draw_payoffs[0:4] = (np.multiply(Q, yi[0:4]))
        draw_payoffs[4:8] = np.multiply((1 - Q), yi[4:8])

    if i < 35:
        new_x = p1_px - np.dot(draw_payoffs, etp_calculation)
        np.place(new_x, new_x < 0, 0)
        np.place(new_x, new_x > 1, 1)

        upper_part_Q = np.sum(np.multiply(yi[4:8],
                                          new_x[:, 4:8]))
        leftdown_part_Q = np.sum(
            np.multiply(yi[0:4], (1 - new_x[:, 0:4])))
        rightdown_part_Q = np.sum(np.multiply(yi[4:8],
                                              new_x[:, 4:8]))

        Q_between = upper_part_Q / (leftdown_part_Q + rightdown_part_Q)
        Q = np.hstack((Q, Q_between))

        draw_payoffs[0:4] = (np.multiply(Q[i], yi[0:4]))
        draw_payoffs[4:8] = np.multiply((1 - Q[i]), yi[4:8])

        print(draw_payoffs)
        print(np.sum(draw_payoffs))