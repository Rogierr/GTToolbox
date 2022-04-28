import numpy as np
import matplotlib.pyplot as plt
from computation.random_strategy_draw import random_strategy_draw
from game.game import RepeatedGame
from FD_functions.learning_curves import scurve

points = 1000000

# initialize the payoff matrices
# p1_payoff_PD = np.array([[6, 3], [8, 4]])
# p2_payoff_PD = np.array([[6, 8], [3, 4]])
#
# Prisoners_Dilemma = RepeatedGame(p1_payoff_PD, p2_payoff_PD)
# Prisoners_Dilemma.plot_single_period_pure_rewards()
# Prisoners_Dilemma.plot_all_rewards(points)
# Prisoners_Dilemma.compute_threat_point(100000, True, True, True)
# Prisoners_Dilemma.plot_pareto_rewards(points)
# Prisoners_Dilemma.plot_threat_point()
# Prisoners_Dilemma.plot_threat_point_lines()
# plt.tight_layout()
# plt.savefig('./figures/PD_Basic_Game.png')
# plt.show()
#
p1_payoff_SH = np.array([[8, 3], [6, 4]])
p2_payoff_SH = np.array([[8, 6], [3 ,4]])

Stag_Hunt = RepeatedGame(p1_payoff_SH, p2_payoff_SH)
Stag_Hunt.plot_single_period_pure_rewards()
Stag_Hunt.plot_all_rewards(points)
Stag_Hunt.plot_pareto_rewards(points)
Stag_Hunt.compute_threat_point(100000, True, True, True)
Stag_Hunt.plot_threat_point()
Stag_Hunt.plot_threat_point_lines()
plt.tight_layout()
plt.savefig('./figures/SH_Basic_Game_with_lines.png')
plt.show()
