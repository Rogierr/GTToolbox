import time
import numpy as np
from scipy.spatial import ConvexHull  # import scipy convex hull package
import matplotlib.pyplot as plt  # import package to plot stuff

from computation.balance_equation_all import balance_equation_all
from computation.random_strategy_draw import random_strategy_draw
from FD_functions.fd_function import fd_function
from FD_functions.mu_function import mu_function
from FD_functions.rho_function import rho_function
from FD_functions.profit_function import profit_function

__all__ = ['plot_convex_hull_pure_rewards', 'plot_single_period_pure_rewards', 'plot_all_rewards']


def plot_convex_hull_pure_rewards(game):
    """Here we plot a convex hull around the pure reward point, therefore resulting
    in the total possible reward space"""

    all_payoffs = np.array([game.payoff_p1_merged, game.payoff_p2_merged])  # create one array of payoffs
    all_payoffs = np.transpose(all_payoffs)  # and rotate this one

    rewards_convex_hull = ConvexHull(all_payoffs)  # retain the convex hull of the payoffs
    plt.fill(all_payoffs[rewards_convex_hull.vertices, 0], all_payoffs[rewards_convex_hull.vertices, 1], color='k')
    # here above we fill the convex hull in black


def plot_single_period_pure_rewards(self):
    "Here we plot the pure rewards possible for a single period"

    plt.figure()  # create a figure
    payoff_p1_g1_flat = self.payoff_p1_game1.A1  # create a flattend payoff of p1 in game 1
    payoff_p2_g1_flat = self.payoff_p2_game1.A1  # create a flattend payoff of p2 in game 1
    plt.scatter(payoff_p1_g1_flat, payoff_p2_g1_flat, label="Pure reward points Game 1",
                zorder=15)  # plot payoffs game 1

    payoff_p1_g2_flat = self.payoff_p1_game2.A1  # create a flattend payoff of p1 in game 2
    payoff_p2_g2_flat = self.payoff_p2_game2.A1  # and for p2 in game 2
    plt.scatter(payoff_p1_g2_flat, payoff_p2_g2_flat, label="Pure reward points Game 2",
                zorder=15)  # plotting this again

    plt.xlabel("Payoff Player 1")  # giving the x-axis the label of payoff p1
    plt.ylabel("Payoff Player 2")  # and the payoff of the y-axis is that of p2
    plt.title("Reward points of ETP game")  # and we give it a nice titel
    plt.legend()
    plt.show()

    
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

    print(np.nanmax(payoffs_p1))
    store = np.where(np.nanmax(payoffs_p1))
    print(draw_payoffs[store[0],:])

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

    plt.figure()
    plt.title("Small Fish Wars with Hysteresis and Rarity Value")
    plt.xlabel("Rewards player 1")
    plt.ylabel("Rewards player 2")
    plt.scatter(payoffs_p1, payoffs_p2, s=0.3)
    #         plt.fill(all_payoffs[Convex_Hull_Payoffs.vertices,0],all_payoffs[Convex_Hull_Payoffs.vertices,1],color='y', zorder=5, label="Obtainable rewards")
    end_time = time.time()
    plt.show()

    print("Total time taken to plot all reward points:", end_time - start_time)
