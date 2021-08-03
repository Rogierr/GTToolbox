import numpy as np
import matplotlib.pyplot as plt
from computation.random_strategy_draw import random_strategy_draw
from game.game import RepeatedGame
from FD_functions.learning_curves import scurve

points = 1000000

# initialize the payoff matrices
p1_payoff_PD = np.matrix('6 3; 8 4')
p2_payoff_PD = np.matrix('6 8; 3 4')

Prisoners_Dilemma = RepeatedGame(p1_payoff_PD, p2_payoff_PD)

p1_payoff_SH = np.matrix('8 3; 6 4')
p2_payoff_SH = np.matrix('8 6; 3 4')

Stag_Hunt = RepeatedGame(p1_payoff_SH, p2_payoff_SH)
#
# Prisoners_Dilemma.plot_all_rewards(points, "Prisoners Dilemma - Basic Data Sharing Dilemma")
# Prisoners_Dilemma.compute_threat_point(100000, True, True, True)
# Prisoners_Dilemma.plot_threat_point(0)
# Prisoners_Dilemma.plot_threat_point_lines(0)

# Stag_Hunt.plot_all_rewards(points, "Stag Hunt - Basic Data Sharing Dilemma")
# Stag_Hunt.compute_threat_point(100000, True, True, True)
# Stag_Hunt.plot_threat_point(0)
# Stag_Hunt.plot_threat_point_lines(0)

for i in range(0, 11):
    Prisoners_Dilemma.set_phi(i*0.1+0)
    # Prisoners_Dilemma.define_learning_curve('learning')
    # Prisoners_Dilemma.define_mu('sb')
    # Prisoners_Dilemma.plot_all_rewards(points, 'Prisoners Dilemma Singular Benefits with phi:' + str(i*0.1+0))
    # print("Threat point PD tanh sb")
    # Prisoners_Dilemma.compute_threat_point(100000, True, True, True)
    # Prisoners_Dilemma.plot_threat_point(i)
    # Prisoners_Dilemma.plot_threat_point_lines(i)


    # Prisoners_Dilemma.define_learning_curve('learning')
    # Prisoners_Dilemma.define_mu('mb')
    # Prisoners_Dilemma.plot_all_rewards(points, 'Prisoners Dilemma Mutual Benefits with phi:' + str(i*0.1+0))
    # print("Threat point PD tanh MB")
    # Prisoners_Dilemma.compute_threat_point(100000, True, True, True)
    # Prisoners_Dilemma.plot_threat_point(i)
    # Prisoners_Dilemma.plot_threat_point_lines(i)

    #
    Stag_Hunt.set_phi(i*0.1+0)
    Stag_Hunt.define_learning_curve('learning')
    Stag_Hunt.define_mu('sb')
    Stag_Hunt.plot_all_rewards(points, 'Stag Hunt Singular Benefits with phi:' + str(i*0.1+0))
    print("Threat point SH tanh SB")
    Stag_Hunt.compute_threat_point(100000, True, True, True)
    Stag_Hunt.plot_threat_point(i)
    Stag_Hunt.plot_threat_point_lines(i)

    # Stag_Hunt.define_learning_curve('learning')
    # Stag_Hunt.define_mu('mb')
    # Stag_Hunt.plot_all_rewards(points, 'Stag Hunt Mutual Benefits with phi:' + str(i*0.1+0))
    # print("Threat point SH tanh MB")
    # Stag_Hunt.compute_threat_point(100000, True, True, True)
    # Stag_Hunt.plot_threat_point(i)
    # Stag_Hunt.plot_threat_point_lines(i)

