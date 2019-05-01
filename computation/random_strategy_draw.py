import numpy as np


def random_strategy_draw(points, number_of_actions):
    """This function draws random strategies from a beta distribution, based on the number of points and actions"""

    # draw some strategies and normalize them
    strategies_drawn = np.random.beta(0.5, 0.5, (points, number_of_actions))
    strategies_drawn = np.divide(strategies_drawn, np.sum(strategies_drawn, axis=1).reshape([points, 1]))

    return strategies_drawn
