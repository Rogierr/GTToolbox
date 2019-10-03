import numpy as np
import random

game_p1 = np.array([3, 2])  # we have reduced the minimax example to this game

# we initialize the following values
Q = np.ones((2))        # Q-table
alpha = 0.999           # learning rate
gamma = 0.99            # discount factor
epsilon = 0.2           # epsilon for the boundary of exploring or exploiting


def determine_action():
    """
    This function chooses to explore or exploit
    :return:
    """

    if random.uniform(0, 1) < epsilon:  # if the sample drawn is lower than epsilon, we explore
        if random.uniform(0, 1) < 0.5:
            action = 0
        else:
            action = 1
    else:   # else we exploit
        if np.argmax(Q) == 0:
            action = 0
        else:
            action = 1

    return action

iterations = 3000   # total number of iterations


for i in range(0, iterations):
    # here is the general loop
    print("Now at iteration", i+1)
    action = determine_action() # we select an action
    print("Player will now play action:", action)

    reward = game_p1[action]    # get the reward
    print("Current reward in the game is:", reward)
    Q[action] = Q[action] + alpha * (reward + gamma * np.max(Q) - Q[action])    # update the Q-table
    print("New update to the Q table makes it: \n", Q)
    print("Currently at alpha", alpha)
    print("End of Q-learning")
    print("")
    print("")