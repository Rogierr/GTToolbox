import numpy as np

def payoffs_sorted(points, payoffs, actions):
    "This function sorts the payoffs in order to prepare the threat point"

    # create ranges for points and actions
    points_range = np.arange(points)
    actions_range = np.arange(actions)

    payoffs_sort = np.zeros((points, actions))  # nitialize the payoffs sort

    # sort the payoffs!
    for x in np.nditer(points_range):
        for i in np.nditer(actions_range):
            payoffs_sort[x, i] = payoffs[points * i + x]

    return payoffs_sort