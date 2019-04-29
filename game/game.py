import numpy as np


class ETPGame:
    """The ETP Game class represents the Type III games from the thesis, with or without ESP."""

    def __init__(self, payoff_p1_game1, payoff_p2_game1, payoff_p1_game2, payoff_p2_game2, trmatrixg1, trmatrixg2,
                 trmatrixg3, trmatrixg4, matrixa):
        """Here below we initialize the game by storing payoff and transition matrices according to the upper input."""

        # here below we just store some values that are being put in
        self.payoff_p1_game1 = payoff_p1_game1  # payoff p1 in game 1
        self.payoff_p2_game1 = payoff_p2_game1  # payoff p2 in game 1

        self.payoff_p1_game2 = payoff_p1_game2  # payoff p1 in game 2
        self.payoff_p2_game2 = payoff_p2_game2  # payoff p2 in game 2

        self.transition_matrix_game1_to1 = trmatrixg1  # transition matrix from game 1 to game 1
        self.transition_matrix_game2_to1 = trmatrixg2  # transition matrix from game 2 to game 1

        self.transition_matrix_game1_to2 = trmatrixg3  # transition matrix from game 1 to game 2
        self.transition_matrix_game2_to2 = trmatrixg4  # transition matrix from game 2 to game 2
        # here below we just store some values that are being put in

        # some adjustments on the size of the games
        self.payoff_p1_g1_flat = self.payoff_p1_game1.A1  # store the flatten payoff of p1 game 1
        self.payoff_p2_g1_flat = self.payoff_p2_game1.A1  # store the flatten payoff of p2 game 1

        self.payoff_p1_g2_flat = self.payoff_p1_game2.A1  # store the flatten payoff of p1 game 2
        self.payoff_p2_g2_flat = self.payoff_p2_game2.A1  # store the flatten payoff of p2 game 2

        self.payoff_p1_size = self.payoff_p1_g1_flat.size + self.payoff_p1_g2_flat.size
        self.payoff_p2_size = self.payoff_p2_g1_flat.size + self.payoff_p2_g2_flat.size

        self.total_payoffs = self.payoff_p1_game1.size + self.payoff_p2_game2.size

        self.payoff_p1_actions = self.payoff_p1_game1.shape[0] + self.payoff_p1_game2.shape[0]
        self.payoff_p2_actions = self.payoff_p2_game1.shape[1] + self.payoff_p2_game2.shape[1]
        self.total_actions = self.payoff_p1_actions + self.payoff_p2_actions

        self.payoff_p1 = np.zeros(self.payoff_p1_size)
        self.payoff_p1[0:self.payoff_p1_g1_flat.size] = self.payoff_p1_g1_flat
        self.payoff_p1[self.payoff_p1_g1_flat.size:self.payoff_p1_size] = self.payoff_p1_g2_flat

        self.payoff_p2 = np.zeros(self.payoff_p2_size)
        self.payoff_p2[0:self.payoff_p2_g1_flat.size] = self.payoff_p2_g1_flat
        self.payoff_p2[self.payoff_p2_g1_flat.size:self.payoff_p2_size] = self.payoff_p2_g2_flat

        self.trans_matr_game1_to1_flat = self.transition_matrix_game1_to1.flatten()     # flattened trans matrices
        self.trans_matr_game2_to1_flat = self.transition_matrix_game2_to1.flatten()

        self.trans_matr_game1_to2_flat = self.transition_matrix_game1_to2.flatten()
        self.trans_matr_game2_to2_flat = self.transition_matrix_game2_to2.flatten()
        # here above are some adjustments on the size of the games

        # here below we create px
        self.px = np.concatenate([self.trans_matr_game1_to1_flat, self.trans_matr_game2_to1_flat], axis=1)

        self.payoff_p1_merged = np.concatenate((self.payoff_p1_g1_flat, self.payoff_p1_g2_flat))  # merge p1 payoffs
        self.payoff_p2_merged = np.concatenate((self.payoff_p2_g1_flat, self.payoff_p2_g2_flat))  # merge p2 payoffs

        self.etp_matrix = matrixa

        # we just set some things to false a default initialization
        self.hysteresis = False
        self.FD = False
        self.FD_function_use = "FD"
        self.mu = False
        self.plotting_rarity = False
        self.rarity = False
        self.printing = False  # set printing to False
        self.phi = 0
        self.m = 0

        self.best_pure_strategies = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]])

    def activate_fd(self):
        print("Game is now an ESP game")

        self.FD = True

    def activate_hysteresis(self, phi):
        print("Hysteresis is now active")

        self.hysteresis = True
        self.phi = phi

    def activate_rarity(self):
        print("Rarity function active")

        self.rarity = True

    def deactivate_fd(self):
        print("FD function deactivated")
        self.FD = False

    def deactivate_hysteresis(self):
        print("Hysteresis disabled")

        self.hysteresis = False

    def deactivate_rarity(self):
        print("Rarity function deactivated")

        self.rarity = False

    def adjust_fd(self, type_function):
        if type_function == "mu":
            self.FD_function_use = "mu"
            self.m = 0
        else:
            self.FD_function_use = "FD"

    def adjust_mu(self, m):
        self.m = m

        print("Mu adjusted, now has:")
        print("M = ", m)

