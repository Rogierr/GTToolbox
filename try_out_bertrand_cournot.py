from game.bertrand_cournot import ETPGame
import matplotlib.pyplot as plt

game = ETPGame(0)
game.optimal_profit(True, True, 100000, 100000)
game.plot_equilibrium_outcomes(0)
game.check_extremes()
# plt.tight_layout()
plt.savefig('./figures/collusion/test.png')
plt.show()