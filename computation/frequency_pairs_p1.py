import numpy as np


def frequency_pairs_p1(self, points, strategies_drawn):
    "This function sorts the strategies based on the responses"

    # store the game size
    game_size_1 = self.payoff_p1_game1.size
    game_size_2 = self.payoff_p1_game2.size

    # store the actions of p1 in both game
    p1_actions_game1 = self.payoff_p1_game1.shape[0]
    p1_actions_game2 = self.payoff_p1_game2.shape[0]

    p1_actions_combi = p1_actions_game1 * p1_actions_game2
    p1_action_range = np.arange(p1_actions_combi)

    # initialize frequency pairs
    frequency_pairs = np.zeros((points * (p1_actions_game1 * p1_actions_game2), game_size_1 + game_size_2))

    # set the range for both games
    p1_act_game1_range = np.arange(p1_actions_game1)
    p1_act_game2_range = np.arange(p1_actions_game2)

    # create best response for game 1
    for i in np.nditer(p1_action_range):
        for j in np.nditer(p1_act_game1_range):
            mod_remain = np.mod(i, p1_actions_game1)
            frequency_pairs[i * points:(i + 1) * points, p1_actions_game1 * mod_remain + j] = strategies_drawn[
                                                                                              :, j]

    # loop for best responses for game 2
    for i in np.nditer(p1_action_range):
        for j in np.nditer(p1_act_game2_range):
            floor_div = np.floor_divide(i, p1_actions_game2)
            frequency_pairs[i * points:(i + 1) * points,
            j + game_size_1 + (p1_actions_game1 * floor_div)] = strategies_drawn[:, p1_actions_game1 + j]

    return frequency_pairs