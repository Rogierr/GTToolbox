def plot_threat_point(self):
    "This function plots the threat point if found"
    plt.scatter(self.threat_point[0], self.threat_point[1], zorder=10, color='r', label='Threat point')
    plt.legend()