import numpy as np
from game.game import ETPGame
import matplotlib.pyplot as plt


p1_1 = np.matrix('16 14; 28 26')
p2_1 = np.matrix('16 28; 14 26')

p1_2 = np.matrix('4 3.5; 7 13')
p2_2 = np.matrix('4 7; 3.5 13')

ts1_1 = np.matrix('16 14; 28 24')
ts2_1 = np.matrix('16 28; 14 24')

ts1_2 = np.matrix('4 3.5; 7 6')
ts2_2 = np.matrix('4 7; 3.5 6')

trans1_1 = np.matrix('0.8 0.7; 0.7 0.6')
trans2_1 = np.matrix('0.5 0.4; 0.4 0.15')

trans1_2 = np.matrix('0.2 0.3; 0.3 0.4')
trans2_2 = np.matrix('0.5 0.6; 0.6 0.85')

matrixA = np.matrix('0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.35 0.3 0.3 0.25 0.2 0.15 0.15 0.05; 0.35 0.3 0.3 0.25 0.2 0.15 0.15 0.05; 0.7 0.6 0.6 0.5 0.4 0.3 0.3 0.1; 0 0 0 0 0 0 0 0; 0.35 0.3 0.3 0.25 0.2 0.15 0.15 0.05; 0.35 0.3 0.3 0.25 0.2 0.15 0.15 0.05; 0.7 0.6 0.6 0.5 0.4 0.3 0.3 0.1')
matrixB = np.matrix('0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.0 0.0 0.0 0.00 0.0 0.0 0.0 0.00; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0 0 0 0 0 0 0 0; 0.0 0.0 0.0 0.00 0.0 0.00 0.00 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.00; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0')
matrixC = np.matrix('0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.57 0.5 0.5 0.45 0.39 0.315 0.315 0.12; 0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.57 0.5 0.5 0.45 0.39 0.315 0.315 0.12')

ETP = ETPGame(p1_1,p2_1,p1_2,p2_2,trans1_1,trans2_1,trans1_2,trans2_2,matrixC)
# TestGame = ETPGame(ts1_1,ts2_1,ts1_2,ts2_2, trans1_1,trans2_1, trans1_2, trans2_2, matrixA)

phi_range = np.arange(0, 1.51, 0.01)

for i in np.arange(0, 151):
# ETP.activate_fd()
    print(i)
    ETP.activate_rarity()
    ETP.plotting_rare("Rarity")
    ETP.activate_hysteresis(phi_range[i])
    ETP.adjust_mu(0.05)
    ETP.plot_all_rewards(1000000, i)

# rewards_p1 = np.array([16, 0.56539, 1.13078, -1.9170, 4, 0.3917, 0.7834, 35.213])
# rewards_p2 = np.array([16, 1.13078, 0.56539, -1.9042, 4, 0.7834, 0.3917, 35.213])

# sixteen_p1 = np.array([13.6, 4.946, 9.233, 5.574, 7.801, -2.452, -4.539, -3.556, 8.377, -2.817, -4.904, -3.962, 2.033,
#                        -2.694, -4.252, 1.685])
# sixteen_p2 = np.array([13.6, 9.233, 4.946, 5.574, 8.377, -4.904, -2.817, -3.963, 7.801, -4.539, -2.452, -3.556, 2.033,
#                        -4.252, -2.694, 1.685])
#
# new_p1 = np.array([13.6, 7.719, 8.295, 2.0342, 4.946, -2.452])
# new_p2 = np.array([13.6, 8.295, 7.719, 2.0342, 9.234, -4.904])

# plt.scatter(rewards_p1, rewards_p2)
# # plt.scatter(sixteen_p1, sixteen_p2)
# plt.scatter(new_p1, new_p2)

# ETP.compute_threat_point(100000, True, True, True)
# ETP.compute_maximin(100000, True, True)


# TestGame.activate_fd()
# TestGame.compute_threat_point(500000, True, True, True)
# TestGame.compute_maximin(500000, True, True)
