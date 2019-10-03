import time
import numpy as np
from osbrain import run_nameserver
from osbrain import run_agent
from osbrain import Agent
from scipy.optimize import linprog
import random

game_p1 = np.array([3, 2])

Q = np.ones((2))
alpha = 1
gamma = 0.99
epsilon = 0.2


def determine_action():
    if random.uniform(0, 1) < epsilon:
        if random.uniform(0, 1) < 0.5:
            action = 0
        else:
            action = 1
    else:
        if np.argmax(Q) == 0:
            action = 0
        else:
            action = 1

    return action

iterations = 1000000


for i in range(0, iterations):
    # time.sleep(1)
    print("Now at iteration", i+1)
    action = determine_action()
    print("Player will now play action:", action)
    # opponent_action = 1
    # print("Opponent will now play action:", opponent_action)
    reward = game_p1[action]
    print("Current reward in the game is:", reward)
    Q[action] = Q[action] + alpha * (reward + gamma * np.max(Q) - Q[action])
    print("New update to the Q table makes it: \n", Q)
    print("Currently at alpha", alpha)
    print("End of Q-learning")
    print("")
    print("")