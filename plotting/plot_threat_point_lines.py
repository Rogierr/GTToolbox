def plot_threat_point_lines(self):
    "This function plots lines around the threat point indicating the limits for the NE"

    plt.plot([self.threat_point[0], self.threat_point[0]], [self.threat_point[1], self.maximal_payoffs[1]],
             color='k', zorder=15)
    plt.plot([self.threat_point[0], self.maximal_payoffs[0]], [self.threat_point[1], self.threat_point[1]],
             color='k', zorder=15)