import numpy as np


class ETPGame:
    "The ETP Game class represents the Type III games from the thesis, with or without ESP."

    def __init__(self, payoff_p1_game1, payoff_p2_game1, payoff_p1_game2, payoff_p2_game2, trmatrixg1, trmatrixg2,
                 trmatrixg3, trmatrixg4, matrixa):
        "Here below we initialize the game by storing payoff and transition matrices according to the upper input."
        self.payoff_p1_game1 = payoff_p1_game1  # payoff p1 in game 1
        self.payoff_p2_game1 = payoff_p2_game1  # payoff p2 in game 1

        self.payoff_p1_game2 = payoff_p1_game2  # payoff p1 in game 2
        self.payoff_p2_game2 = payoff_p2_game2  # payoff p2 in game 2

        self.transition_matrix_game1_to1 = trmatrixg1  # transition matrix from game 1 to game 1
        self.transition_matrix_game2_to1 = trmatrixg2  # transition matrix from game 2 to game 1

        self.transition_matrix_game1_to2 = trmatrixg3  # transition matrix from game 1 to game 2
        self.transition_matrix_game2_to2 = trmatrixg4  # transition matrix from game 2 to game 2

        self.payoff_p1_g1_flat = self.payoff_p1_game1.A1  # store the flatten payoff of p1 game 1
        self.payoff_p2_g1_flat = self.payoff_p2_game1.A1  # store the flatten payoff of p2 game 1

        self.payoff_p1_g2_flat = self.payoff_p1_game2.A1  # store the flatten payoff of p1 game 2
        self.payoff_p2_g2_flat = self.payoff_p2_game2.A1  # store the flatten payoff of p2 game 2

        self.payoff_p1_merged = np.concatenate((self.payoff_p1_g1_flat, self.payoff_p1_g2_flat))  # merge p1 payoffs
        self.payoff_p2_merged = np.concatenate((self.payoff_p2_g1_flat, self.payoff_p2_g2_flat))  # merge p2 payoffs

        self.etp_matrix = matrixa

        self.printing = False  # set printing to False

        self.best_pure_strategies = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
