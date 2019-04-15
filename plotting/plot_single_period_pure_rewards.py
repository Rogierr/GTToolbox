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