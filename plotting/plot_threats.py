import matplotlib.pyplot as plt
import pandas as pd

__all__ = ['plot_threat_point', 'plot_threat_point_lines']


def plot_threat_point(self, k):
    """This function plots the threat point if found"""

    # plt.figure()
    plt.scatter(self.threat_point[0], self.threat_point[1], zorder=10, color='r', label='Threat point')
    plt.legend()
    # plt.savefig('figures/m = 1, phi = 0, with threat.png', dpi=300, bbox_inches="tight")
    # plt.show()


def plot_threat_point_lines(self, k):
    "This function plots lines around the threat point indicating the limits for the NE"

    # plt.figure()
    plt.plot([self.threat_point[0], self.threat_point[0]], [self.threat_point[1], self.maximal_payoffs[1]],
             color='k', zorder=15)
    plt.plot([self.threat_point[0], self.maximal_payoffs[0]], [self.threat_point[1], self.threat_point[1]],
             color='k', zorder=15)
    plt.xlim([2,16])
    plt.ylim([2,16])
    plt.savefig('figures/SH_SB_phi_%d.png'%k, dpi=300, bbox_inches="tight")
    plt.show()

    if self.dataframe is None:
        self.dataframe = pd.DataFrame(columns=['minrew0', 'minrew1', 'tp0', 'tp1', 'maxrew0', 'maxrew1'])
        self.dataframe['minrew0'] = pd.Series(self.minimal_payoffs[0])
        self.dataframe['minrew1'] = self.minimal_payoffs[1]
        self.dataframe['tp0'] = self.threat_point[0]
        self.dataframe['tp1'] = self.threat_point[1]
        self.dataframe['maxrew0'] = self.maximal_payoffs[0]
        self.dataframe['maxrew1'] = self.maximal_payoffs[1]

        self.dataframe_pareto = pd.DataFrame(self.pareto_rewards)

    else:
        self.dataframe.loc[k] = [self.minimal_payoffs[0], self.minimal_payoffs[1], self.threat_point[0], self.threat_point[1],
                                  self.maximal_payoffs[0], self.maximal_payoffs[1]]

        print(self.dataframe)

        self.dataframe_pareto = pd.concat([self.dataframe_pareto, pd.DataFrame(self.pareto_rewards)], axis=1)

        print(self.dataframe_pareto)

    if k == 10:
        self.dataframe.to_csv('./data/sh_mb.csv')
        self.dataframe_pareto.to_csv('./data/sh_mb_pareto.csv')
