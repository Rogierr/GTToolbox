# General packages
import time
import numpy as np
import matplotlib.pyplot as plt

# GTToolbox functions
from computation.balance_equation_all import balance_equation_all
from computation.random_strategy_draw import random_strategy_draw
# from scipy.spatial import ConvexHull


def plot_all_rewards(self, points):
    print("Now plotting all rewards")

    start_time = time.time()  # timer start

    draw_payoffs = random_strategy_draw(points, self.payoff_p1_actions)   # draw some payoffs

    # calculate the balance equations
    draw_payoffs = balance_equation_all(self, points, self.payoff_p1_g1_flat.size, self.payoff_p1_actions, draw_payoffs)

    fd = 1  # initialize value in case there is no FD function

    # activate the FD function
    if self.FD:
        if self.FD_function_use == "FD":
            fd = self.FD_function(draw_payoffs)
        elif self.FD_function_use == "mu":
            fd = self.mu_function(self.rho_function(draw_payoffs))

    payoffs_p1 = np.sum(np.multiply(draw_payoffs, self.payoff_p1), axis=1)
    payoffs_p2 = np.sum(np.multiply(draw_payoffs, self.payoff_p2), axis=1)

    if self.plotting_rarity == "Rarity" and self.rarity:
        print("Plotting with rarity active")
        rho = self.rho_function(draw_payoffs)
        mu = self.mu_function(rho)
        payoffs_p1 = np.multiply(self.profit_function(mu), payoffs_p1)
        payoffs_p2 = np.multiply(self.profit_function(mu), payoffs_p2)
    else:
        print("Normal plotting active")
        payoffs_p1 = np.multiply(fd, payoffs_p1)
        payoffs_p2 = np.multiply(fd, payoffs_p2)

    delete_indic = np.where(np.isnan(payoffs_p1))
    payoffs_p1 = np.delete(payoffs_p1, delete_indic[0], 0)
    payoffs_p2 = np.delete(payoffs_p2, delete_indic[0], 0)

    self.maximal_payoffs = np.zeros(2)
    self.maximal_payoffs = [np.max(payoffs_p1), np.max(payoffs_p2)]

    self.minimal_payoffs = np.zeros(2)
    self.minimal_payoffs = [np.min(payoffs_p1), np.min(payoffs_p2)]

    # all_payoffs = np.array([payoffs_p1, payoffs_p2])
    # all_payoffs = np.transpose(all_payoffs)
    # convex_hull_payoffs = ConvexHull(all_payoffs, qhull_options='QbB')

    plt.figure()
    plt.scatter(payoffs_p1, payoffs_p2)
    # plt.fill(all_payoffs[Convex_Hull_Payoffs.vertices,0],all_payoffs[Convex_Hull_Payoffs.vertices,1],color='y',
    # zorder=5, label="Obtainable rewards")
    plt.show()

    end_time = time.time()

    print("Total time taken to plot all reward points:", end_time - start_time)
