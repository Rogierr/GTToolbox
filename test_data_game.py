import numpy as np
import matplotlib.pyplot as plt
from computation.random_strategy_draw import random_strategy_draw

p1_payoff = np.matrix('6 0; 8 1')
p2_payoff = np.matrix('6 8; 0 1')

play_frequencies = random_strategy_draw(1000000, p1_payoff.size)

rewards_p1 = np.sum(np.multiply(p1_payoff.A1, play_frequencies), axis=1)
rewards_p2 = np.sum(np.multiply(p2_payoff.A1, play_frequencies), axis=1)

plt.scatter(rewards_p1, rewards_p2)
plt.show()

