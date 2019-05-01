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
x = np.array([0, 0, 0, 1, 0, 0, 0, 0])
# ETP.activate_fd()
ETP.activate_rarity()
ETP.plotting_rare("Rarity")
# ETP.activate_hysteresis(2.5)
ETP.adjust_mu(0.05)
ETP.plot_all_rewards(5000000)

rewards_p1 = np.array([16, 0.56539, 1.13078, -1.9170, 4, 0.3917, 0.7834, 35.213])
rewards_p2 = np.array([16, 1.13078, 0.56539, -1.9042, 4, 0.7834, 0.3917, 35.213])
plt.scatter(rewards_p1, rewards_p2)
plt.show()
# ETP.compute_threat_point(100000, True, True, True)
# ETP.compute_maximin(100000, True, True)


# TestGame.activate_fd()
# TestGame.compute_threat_point(500000, True, True, True)
# TestGame.compute_maximin(500000, True, True)
