import time
import numpy as np
from osbrain import run_nameserver
from osbrain import run_agent
from osbrain import Agent
from scipy.optimize import linprog
import random

game_p1 = np.array([[-1, -10], [0, -9]])
game_p2 = np.array([[-1, 0], [-10, -9]])

Q = np.ones((2, 2))
V = 1
pi = np.ones(2) / 2
pi_opponent = np.ones(2) / 2
alpha = 1
epsilon = 0.2
gamma = 1

current_state = 0
next_state = 0


def determine_action(pi):
    if random.uniform(0, 1) < epsilon:
        if random.uniform(0, 1) < 0.5:
            action = 0
        else:
            action = 1
    else:
        if random.uniform(0, 1) < pi[0]:
            action = 0
        else:
            action = 1

    return action

def random_player():
    if random.uniform(0, 1) < 0.5:
        action = 0
    else:
        action = 1

    return action

def updatePolicy(Q, pi):
    c = np.zeros(3)
    c[0] = -1
    A_ub = np.ones((2, 3))
    A_ub[:, 1:] = -Q.T
    b_ub = np.zeros(2)
    A_eq = np.ones((1, 3))
    A_eq[0, 0] = 0
    b_eq = [1]
    bounds = ((None, None), (0, 1), (0, 1))

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    if res.success:
        pi = res.x[1:]
    else:
        print("Alert : %s" % res.message)
        pi = pi


    return pi

iterations = 10000

for i in range(0, iterations):
    print("Now at iteration", i+1)
    action = determine_action(pi)
    print("Player will now play action:", action)
    opponent_action = 1
    print("Opponent will now play action:", opponent_action)
    reward = game_p1[action, opponent_action]
    print("Current reward in the game is:", reward)
    Q[action, opponent_action] = (1 - alpha) * Q[action, opponent_action] + alpha * (reward + gamma + V)
    print("New update to the Q table makes it: \n", Q)
    pi = updatePolicy(Q, pi)
    print("New update to pi results in pi:", pi)
    V = min(np.sum(Q.T * pi, axis=1))
    print("New value of V now is", V)
    print("End of minimax Q-learning")
    print("")
    print("")