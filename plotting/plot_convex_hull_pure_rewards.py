import numpy as np
from scipy.spatial import ConvexHull  # import scipy convex hull package
import matplotlib.pyplot as plt  # import package to plot stuff


def plot_convex_hull_pure_rewards(game):
    """Here we plot a convex hull around the pure reward point, therefore resulting
    in the total possible reward space"""

    all_payoffs = np.array([game.payoff_p1_merged, game.payoff_p2_merged])  # create one array of payoffs
    all_payoffs = np.transpose(all_payoffs)  # and rotate this one

    rewards_convex_hull = ConvexHull(all_payoffs)  # retain the convex hull of the payoffs
    plt.fill(all_payoffs[rewards_convex_hull.vertices, 0], all_payoffs[rewards_convex_hull.vertices, 1], color='k')
    # here above we fill the convex hull in black
