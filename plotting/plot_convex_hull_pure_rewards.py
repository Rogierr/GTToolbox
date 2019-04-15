def plot_convex_hull_pure_rewards(self):
    "Here we plot a convex hull around the pure reward point, therefore resulting in the total possible reward space"

    payoff_p1_g1_flat = self.payoff_p1_game1.A1  # store the flattend payoff of p1 game 1
    payoff_p2_g1_flat = self.payoff_p2_game1.A1  # store the flattend payoff of p2 game 1

    payoff_p1_g2_flat = self.payoff_p1_game2.A1  # store the flattend payoff of p1 game 2
    payoff_p2_g2_flat = self.payoff_p2_game2.A1  # store the flattend payoff of p2 game 2

    payoff_p1_merged = np.concatenate((payoff_p1_g1_flat, payoff_p1_g2_flat))  # merge p1 payoffs
    payoff_p2_merged = np.concatenate((payoff_p2_g1_flat, payoff_p2_g2_flat))  # merge p2 payoffs

    all_payoffs = np.array([payoff_p1_merged, payoff_p2_merged])  # create one array of payoffs
    all_payoffs = np.transpose(all_payoffs)  # and rotate this one

    rewards_convex_hull = ConvexHull(all_payoffs)  # retain the convex hull of the payoffs
    plt.fill(all_payoffs[rewards_convex_hull.vertices, 0], all_payoffs[rewards_convex_hull.vertices, 1], color='k')
    # here above we fill the convex hull in black