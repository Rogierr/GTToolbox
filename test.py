import numpy as np
from game.game import ETPGame
from game_options.activate_fd import activate_fd
from plotting.plot_single_period_pure_rewards import plot_single_period_pure_rewards
from plotting.plot_all_rewards import plot_all_rewards
from plotting.plot_threat_point import plot_threat_point
from threat_point.maximin import optimized_maximin
from threat_point.threat_point import threat_point_optimized

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

FirstTryETP = ETPGame(p1_1,p2_1,p1_2,p2_2,trans1_1,trans2_1,trans1_2,trans2_2,matrixA)
plot_all_rewards(FirstTryETP, 10000000)
# optimized_maximin(FirstTryETP, 10000, True, True)
# threat_point_optimized(FirstTryETP, 10000, True, True, True)