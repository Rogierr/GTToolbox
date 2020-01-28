import numpy as np


def frequency_pairs_p2(self, points, p2_actions, p1_actions, strategies_drawn):
    "Create frequency pairs for P2 based on best responses"

    # store the size of the games
    game_size_1 = self.payoff_p2_game1.size

    if self.class_games == 'ETP':
        game_size_2 = self.payoff_p2_game2.size

    p2_actions_game1 = self.payoff_p2_actions

    if self.class_games == 'ETP':
        p2_actions_game2 = self.payoff_p2_game2.shape[1]

    p2_actions_combo = p2_actions_game1

    if self.class_games == 'ETP':
        p2_actions_combo = p2_actions_game1 * p2_actions_game2

    p2_action_range = np.arange(p2_actions_combo)

    frequency_pairs = np.zeros((points * p2_actions_game1, game_size_1))

    if self.class_games == 'ETP':
        # initialize frequency pairs
        frequency_pairs = np.zeros((points * (p2_actions_game1 * p2_actions_game2), game_size_1 + game_size_2))

    # loop over the first game
    for i in np.nditer(np.arange(p2_actions_game1)):
        for j in np.nditer(p2_action_range):
            modul = np.mod(j, p2_actions_game1)
            frequency_pairs[j * points:(j + 1) * points, p2_actions_game1 * i + modul] = strategies_drawn[:, i]

    if self.class_games == 'ETP':

        # loop over the second game
        for i in np.nditer(np.arange(p2_actions_game2)):
            for j in np.nditer(p2_action_range):
                divide = np.floor_divide(j, p2_actions_game2)
                frequency_pairs[j * points:(j + 1) * points,
                p2_actions_combo + divide + (i * p2_actions_game2)] = strategies_drawn[:, i + p2_actions_game1]

    return frequency_pairs