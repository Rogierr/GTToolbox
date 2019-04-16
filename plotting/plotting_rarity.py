def plotting_rarity(self, plot):
    if plot == "Rarity":
        self.plotting_rarity = plot
        self.m = 1
    elif plot == "Revenue":
        self.plotting_rarity = plot
        self.m = 1
    else:
        self.plotting_rarity = False

    print("Plotting rarity is now:", self.plotting_rarity)