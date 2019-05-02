import matplotlib.pyplot as plt

__all__ = ['plot_threat_point', 'plot_threat_point_lines']


def plot_threat_point(self):
    """This function plots the threat point if found"""

    plt.figure()
    plt.scatter(self.threat_point[0], self.threat_point[1], s=0.3, zorder=10, color='r', label='Threat point')
    plt.legend()
    plt.show()


def plot_threat_point_lines(self):
    "This function plots lines around the threat point indicating the limits for the NE"

    plt.figure()
    plt.plot([self.threat_point[0], self.threat_point[0]], [self.threat_point[1], self.maximal_payoffs[1]],
             color='k', zorder=15)
    plt.plot([self.threat_point[0], self.maximal_payoffs[0]], [self.threat_point[1], self.threat_point[1]],
             color='k', zorder=15)
    plt.show()