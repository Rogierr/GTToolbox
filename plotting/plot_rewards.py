import time
import numpy as np
from scipy.spatial import ConvexHull  # import scipy convex hull package
import matplotlib.pyplot as plt  # import package to plot stuff
from computation.compute_rewards import compute_rewards


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
    """Here we plot the pure rewards possible for a single period"""

    plt.figure()  # create a figure

    if self.type == "RepeatedGame":
        payoffs_p1_flat = self.payoffs_p1.flatten()
        payoffs_p2_flat = self.payoffs_p2.flatten()

        plt.scatter(payoffs_p1_flat, payoffs_p2_flat, label="Pure rewards points", zorder=15)
    else:

        payoff_p1_g1_flat = self.payoff_p1_game1.A1  # create a flattend payoff of p1 in game 1
        payoff_p2_g1_flat = self.payoff_p2_game1.A1  # create a flattend payoff of p2 in game 1
        plt.scatter(payoff_p1_g1_flat, payoff_p2_g1_flat, label="Pure reward points Game 1",
                    zorder=15)  # plot payoffs game 1

        payoff_p1_g2_flat = self.payoff_p1_game2.A1  # create a flattend payoff of p1 in game 2
        payoff_p2_g2_flat = self.payoff_p2_game2.A1  # and for p2 in game 2
        plt.scatter(payoff_p1_g2_flat, payoff_p2_g2_flat, label="Pure reward points Game 2",
                    zorder=15)  # plotting this again

    plt.xlabel("Reward Player 1")  # giving the x-axis the label of payoff p1
    plt.ylabel("Reward Player 2")  # and the payoff of the y-axis is that of p2
    plt.title("Reward points of {0}".format(self.type))  # and we give it a nice titel
    plt.legend()
    plt.show()


def plot_all_rewards(self, points: int):
    print("Now plotting all rewards")

    start_time = time.time()  # timer start

    ## Build a draw function

    payoffs_p1, payoffs_p2 = compute_rewards(self, points)

    self.maximal_payoffs = np.zeros(2)
    self.maximal_payoffs = [np.max(payoffs_p1), np.max(payoffs_p2)]

    self.minimal_payoffs = np.zeros(2)
    self.minimal_payoffs = [np.min(payoffs_p1), np.min(payoffs_p2)]

    # all_payoffs = np.array([payoffs_p1, payoffs_p2])
    # all_payoffs = np.transpose(all_payoffs)
    # Convex_Hull_Payoffs = ConvexHull(all_payoffs, qhull_options='QbB')

    plt.figure()
    plt.title("Total rewards")
    plt.xlabel("Rewards player 1")
    plt.ylabel("Rewards player 2")
    plt.scatter(payoffs_p1, payoffs_p2, s=0.3)

    plt.axis('equal')

    # plt.fill(all_payoffs[Convex_Hull_Payoffs.vertices,0],all_payoffs[Convex_Hull_Payoffs.vertices,1],color='y', zorder=5, label="Obtainable rewards")
    end_time = time.time()

    print("Total time taken to plot all reward points:", end_time - start_time)

