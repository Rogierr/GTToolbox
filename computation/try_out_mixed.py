import numpy as np
from computation.random_strategy_draw import random_strategy_draw


def mixed_strategy_try_out():
    p1_1 = random_strategy_draw(1, 2)
    p2_1 = random_strategy_draw(1, 2)

    p1_2 = random_strategy_draw(1, 2)
    p2_2 = random_strategy_draw(1, 2)

    x = np.zeros(8)

    # print(p1)
    # print(p2)

    c = 0
    for i in range(0, 2):
        for j in range(0, 2):
            x[c] = np.multiply(p1_1[:, i], p2_1[:, j])
            c += 1

    for i in range(0, 2):
        for j in range(0, 2):
            x[c] = np.multiply(p1_2[:, i], p2_2[:, j])
            c += 1

    print(x)
    # print(np.sum(p1))
    # print(np.sum(p2))
    print(np.sum(x))

mixed_strategy_try_out()