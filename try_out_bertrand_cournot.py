from game.bertrand_cournot import ETPGame
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 800, 5)

for i in range(0, len(x)):
    game = ETPGame(0, ad_varA=x[i])
    game.optimal_profit(True, True, 250000, 250000)
    game.plot_equilibrium_outcomes(0)
    game.check_extremes()
    plt.xlim([-450, 750])
    plt.ylim([-25, 600])
    plt.tight_layout()
    plt.savefig('./figures/collusion/reg_bert' + str(i) + '.png')
    plt.show()