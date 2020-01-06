import numpy as np
from game.game import ETPGame
import matplotlib.pyplot as plt
from computation.balance_equation_all import balance_equation_all

#
# p1_1 = np.matrix('16 14; 28 26')
# p2_1 = np.matrix('16 28; 14 26')
#
# p1_2 = np.matrix('4 3.5; 7 6.5')
# p2_2 = np.matrix('4 7; 3.5 6.5')
#
# ts1_1 = np.matrix('16 14; 28 24')
# ts2_1 = np.matrix('16 28; 14 24')
#
# ts1_2 = np.matrix('4 3.5; 7 6')
# ts2_2 = np.matrix('4 7; 3.5 6')
#
# trans1_1 = np.matrix('0.8 0.7; 0.7 0.6')
# trans2_1 = np.matrix('0.5 0.4; 0.4 0.15')
#
# trans1_2 = np.matrix('0.2 0.3; 0.3 0.4')
# trans2_2 = np.matrix('0.5 0.6; 0.6 0.85')
#
# matrixA = np.matrix('0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.35 0.3 0.3 0.25 0.2 0.15 0.15 0.05; 0.35 0.3 0.3 0.25 0.2 0.15 0.15 0.05; 0.7 0.6 0.6 0.5 0.4 0.3 0.3 0.1; 0 0 0 0 0 0 0 0; 0.35 0.3 0.3 0.25 0.2 0.15 0.15 0.05; 0.35 0.3 0.3 0.25 0.2 0.15 0.15 0.05; 0.7 0.6 0.6 0.5 0.4 0.3 0.3 0.1')
# matrixB = np.matrix('0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.0 0.0 0.0 0.00 0.0 0.0 0.0 0.00; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0 0 0 0 0 0 0 0; 0.0 0.0 0.0 0.00 0.0 0.00 0.00 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.00; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0')
# matrixC = np.matrix('0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.57 0.5 0.5 0.45 0.39 0.315 0.315 0.12; 0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.57 0.5 0.5 0.45 0.39 0.315 0.315 0.12')
#
# ETP = ETPGame(p1_1,p2_1,p1_2,p2_2,trans1_1,trans2_1,trans1_2,trans2_2,matrixC)
# TestGame = ETPGame(ts1_1,ts2_1,ts1_2,ts2_2, trans1_1,trans2_1, trans1_2, trans2_2, matrixA)

#m_range = np.arange(0.05, 0.07, 0.0025)
# phi_range = np.arange(0, 1.51, 0.01)

# for i in np.arange(0, np.size(m_range)):
#     # ETP.activate_fd()
#     print(i)
ETP.activate_rarity()
# ETP.plotting_rare("Rarity")
ETP.activate_hysteresis(1.5)
ETP.adjust_mu(1)
# ETP.compute_try_out(5000, 5000)
# ETP.compute_threat_point(2000000, True, True, True)
ETP.plot_all_rewards(5000000, 0)
# ETP.plot_threat_point()


    # ETP.compute_maximin(100000, True, True)

# TestGame.activate_fd()
# TestGame.compute_try_out(2000, 2500)
# TestGame.compute_threat_point(1000000, True, True, True)
# TestGame.compute_maximin(1000000, True, True)
#
# x = np.matrix([0, 0, 0, 1, 0, 0, 0, 1])
# x_res = balance_equation_all(ETP, 1, x)
# print(x_res)
#
#
# x_2 = np.matrix([1, 0, 0, 0, 0, 0, 0, 0])
# x_pres = balance_equation_all(ETP, 1, x_2)
# print(x_pres)