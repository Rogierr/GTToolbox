import time
import numpy as np
import matplotlib.pyplot as plt

from accelerators.aitken_delta_squared import aitken_delta_squared
from computation.balance_equation_all import balance_equation_all
from computation.random_strategy_draw import random_strategy_draw
from FD_functions.fd_function import fd_function
from FD_functions.mu_function import mu_function
from FD_functions.rho_function import rho_function
from FD_functions.profit_function import profit_function

def plot_all_rewards(self, points):
    print("Now plotting all rewards")

    start_time = time.time()  # timer start

    ## Build a draw function

    draw_payoffs = random_strategy_draw(points, self.total_payoffs)

    ## Calculate the balance equations

    draw_payoffs = balance_equation_all(self, points, draw_payoffs)

    # End of balance equations

    fd = 1

    # activate the FD function
    if self.FD or self.rarity:
        if self.FD:
            fd = fd_function(draw_payoffs)
        elif self.mu:
            print("Placeholder")
        elif self.rarity:
            fd = mu_function(self, rho_function(draw_payoffs))

    payoffs_p1 = np.sum(np.multiply(draw_payoffs, self.payoff_p1), axis=1)
    payoffs_p2 = np.sum(np.multiply(draw_payoffs, self.payoff_p2), axis=1)

    if self.plotting_rarity == "Rarity":
        payoffs_p1 = np.multiply(fd, payoffs_p1)
        payoffs_p2 = np.multiply(fd, payoffs_p2)
        print("Plotting with rarity active")
        rho = rho_function(draw_payoffs)
        mu = mu_function(self, rho)
        payoffs_p1 = np.multiply(profit_function(mu), payoffs_p1)
        payoffs_p2 = np.multiply(profit_function(mu), payoffs_p2)
    else:
        print("Normal plotting active")
        payoffs_p1 = np.multiply(fd, payoffs_p1)
        payoffs_p2 = np.multiply(fd, payoffs_p2)

    # here below we just randomly throw out some stuff

    delete_indic = np.where(np.isnan(payoffs_p1))
    payoffs_p1 = np.delete(payoffs_p1, delete_indic[0], 0)
    payoffs_p2 = np.delete(payoffs_p2, delete_indic[0], 0)

    self.maximal_payoffs = np.zeros(2)
    self.maximal_payoffs = [np.max(payoffs_p1), np.max(payoffs_p2)]

    self.minimal_payoffs = np.zeros(2)
    self.minimal_payoffs = [np.min(payoffs_p1), np.min(payoffs_p2)]

    all_payoffs = np.array([payoffs_p1, payoffs_p2])
    all_payoffs = np.transpose(all_payoffs)
    # Convex_Hull_Payoffs = ConvexHull(all_payoffs, qhull_options='QbB')

    plt.scatter(payoffs_p1, payoffs_p2, s=0.3)
    #         plt.fill(all_payoffs[Convex_Hull_Payoffs.vertices,0],all_payoffs[Convex_Hull_Payoffs.vertices,1],color='y', zorder=5, label="Obtainable rewards")
    end_time = time.time()
    plt.show()

    print("Total time taken to plot all reward points:", end_time - start_time)
