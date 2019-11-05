import numpy as np
import time
import matplotlib.pyplot as plt

from computation.random_strategy_draw import random_strategy_draw
from FD_functions.rho_function import rho_function
from FD_functions.mu_function import mu_function
from FD_functions.profit_function import profit_function
from game.game import TypeICTP

p1 = np.matrix('16 14; 28 26')
p2 = np.matrix('16 28; 14 26')

tp_results = np.zeros(951)
maximin_results = np.zeros(951)

for m in np.arange(0.05, 1.001, 0.001):

    print(m)
    index_value = int(((m-0.05)/0.001))
    print(index_value)
    print("")

    testgame = TypeICTP(p1, p2, m)

    points = 100000

    start_time = time.time()

    draw_payoffs = random_strategy_draw(points, 2)
    # print(draw_payoffs)

    p1_actions_range = np.arange(2)
    p2_actions_range = np.arange(2)

    frequency_pairs = np.zeros((points*2, 2*2))

    for i in np.nditer(p1_actions_range):
        for j in np.nditer(p2_actions_range):
            frequency_pairs[i*points:points*(i+1),(2*i)+j] = draw_payoffs[:,j]

    rho_matrix = rho_function(frequency_pairs)
    mu_matrix = mu_function(testgame, rho_matrix)
    # print("The result of the mu matrix")
    # print(mu_matrix)

    profit_matrix = profit_function(mu_matrix)
    # print("The result of the profit matrix")
    # print(profit_matrix)

    # monoto_matrix = np.zeros(points*2)
    # monoto_matrix[mu_matrix > 0.366691233367131] = 1
    # # monoto_matrix[monoto_matrix == 0] = 1

    # print(monoto_matrix)

    result_game_p1 = np.dot(frequency_pairs, p1.A1)

    result_game_p1_rarity = np.multiply(result_game_p1, profit_matrix)

    # print("The results of the game for player 1")
    # print(result_game_p1_rarity)

    payoffs_sort = np.zeros((points, 2))
    points_range = np.arange(points)
    actions_range = np.arange(2)


    for x in np.nditer(points_range):
        for i in np.nditer(actions_range):
            payoffs_sort[x,i] = result_game_p1_rarity[points*i+x]

    # print("")
    # print("")
    # print("The sorted payoff matrix, threat point determination should take place on this")
    # print(payoffs_sort)

    # print("")
    # print("")
    # print("Determining the threat point for p1")
    # print(np.min(np.max(payoffs_sort, axis=1)))

    tp_results[index_value] = np.min(np.max(payoffs_sort, axis=1))

    # result_game_p1_rarity_adj = np.multiply(monoto_matrix, result_game_p1_rarity)
    # payoffs_sort_adj = np.zeros((points, 2))
    #
    # for x in np.nditer(points_range):
    #     for i in np.nditer(actions_range):
    #         payoffs_sort_adj[x,i] = result_game_p1_rarity_adj[points*i+x]
    #
    # print(payoffs_sort_adj)
    # payoffs_sort_adj = (payoffs_sort_adj[~np.any(payoffs_sort_adj == 0, axis=1)])
    #
    # print("")
    # print("")
    # print("Determining the adjusted threat point for p1")
    # print(np.min(np.max(payoffs_sort_adj, axis=1)))

    end_time = time.time()

    total_time = end_time - start_time
    # print("")
    # print("In a time of")
    # print(total_time, "seconds")
    #
    # print("")
    # print("Here we construct the maximin version")

    draw_strategies_p1 = random_strategy_draw(points, 2)
    # print(draw_strategies_p1)

    frequency_pairs_p2 = np.zeros((points*2, 2*2))

    for i in np.nditer(p2_actions_range):
        for j in np.nditer(p1_actions_range):
            frequency_pairs_p2[i * points:points * (i + 1), (i + j * 2)] = draw_strategies_p1[:, j]

    # print("Frequency pairs for P2")
    # print(frequency_pairs_p2)

    rho_matrix_p2 = rho_function(frequency_pairs_p2)
    mu_matrix_p2 = mu_function(testgame, rho_matrix_p2)
    # print("The result of the mu matrix")
    # print(mu_matrix_p2)

    profit_matrix_p2 = profit_function(mu_matrix_p2)
    # print("The result of the profit matrix")
    # print(profit_matrix)

    # monoto_matrix_p2 = np.zeros(points*2)
    # monoto_matrix_p2[mu_matrix_p2 > 0.366691233367131] = 1
    # # monoto_matrix_p2[monoto_matrix_p2 == 0] = 0

    # print(monoto_matrix_p2)

    result_game_p1_maxmin = np.dot(frequency_pairs_p2, p1.A1)

    result_game_p1_maxmin_rarity = np.multiply(result_game_p1_maxmin, profit_matrix_p2)

    payoffs_sort_maxmin = np.zeros((points, 2))

    for x in np.nditer(points_range):
        for i in np.nditer(actions_range):
            payoffs_sort_maxmin[x,i] = result_game_p1_maxmin_rarity[points*i+x]

    # print(payoffs_sort_maxmin)
    #
    # print("")
    # print("")
    # print("Determining the maximin result")
    maximin_results[index_value] = np.max(np.min(payoffs_sort_maxmin, axis=1))

    # result_game_p1_maxmin_rarity_adj = np.multiply(monoto_matrix_p2, result_game_p1_maxmin_rarity)
    # print(result_game_p1_maxmin_rarity_adj)

    # payoffs_sort_maxmin_adj = np.zeros((points, 2))
    #
    # for x in np.nditer(points_range):
    #     for i in np.nditer(actions_range):
    #         payoffs_sort_maxmin_adj[x,i] = result_game_p1_maxmin_rarity_adj[points*i+x]
    #
    # print(payoffs_sort_maxmin_adj)
    #
    # payoffs_sort_maxmin_adj = payoffs_sort_maxmin_adj[~np.any(payoffs_sort_maxmin_adj == 0, axis=1)]
    #
    # print(payoffs_sort_maxmin_adj)
    #
    # print("")
    # print("")
    # print("Determining the adjusted maximin result")
    # print(np.max(np.min(payoffs_sort_maxmin_adj, axis=1)))




# print("")
# print("")
# print("Threat point for m = 0.22  hold maximin <= threat point:", (-0.09670462865804476 <= -0.06877801071214927))
# print("Difference between maximin - threatpoint:", (-0.09670462865804476 - 0.06877801071214927))
# print("")
# print("Threat point for m = 0.227597  hold maximin <= threat point:", (-1.4035850626281379 <= -1.0553595651256968))
# print("Difference between maximin - threatpoint:", (-1.4035850626281379 - -1.0553595651256968))
# print("")
# print("Threat point for m = 0.23  hold maximin <= threat point:", (-1.867829030791706 <= -1.449192264491641))
# print("Difference between maximin - threatpoint:", (-1.867829030791706 - -1.449192264491641))
# print("")
# print("Threat point for m = 0.23  hold maximin <= threat point:", (-1.867829030791706 <= -1.449192264491641))
# print("Difference between maximin - threatpoint:", (-1.867829030791706 - -1.449192264491641))
# print("")
#
# print("Threat point for m = 0.36  hold maximin <= threat point:", (-0.0051440321392467086 <= -0.005144032139246709))
# print("Difference between maximin - threatpoint:", (-0.0051440321392467086 - -0.005144032139246709))
# print("")
# print("Threat point for m = 0.366691  hold maximin <= threat point:", (0.13442117543000953 <= 0.1344211754300095))
# print("Difference between maximin - threatpoint:", (0.13442117543000953 - 0.1344211754300095))
# print("")
# print("Threat point for m = 0.37  hold maximin <= threat point:", (0.2035420107210617 <= 0.20354201072106165))
# print("Difference between maximin - threatpoint:", (0.2035420107210617 - 0.20354201072106165))
# print("")
#
# print("Threat point for m = 0.68  hold maximin <= threat point:", (6.88430085189859 <= 6.884300851898589))
# print("Difference between maximin - threatpoint:", (6.88430085189859 - 6.884300851898589))
# print("")
# print("Threat point for m = 0.680123  hold maximin <= threat point:", (6.887007556777468 <= 6.8870075567774665))
# print("Difference between maximin - threatpoint:", (6.887007556777468 - 6.8870075567774665))
# print("")
# print("Threat point for m = 0.69  hold maximin <= threat point:", (7.104455477013581 <= 7.104455477013574))
# print("Difference between maximin - threatpoint:", (7.104455477013581 - 7.104455477013574))
# print("")
# print(tp_results)

abs_diff = abs(maximin_results - tp_results)
np.savetxt("threat_points_func5.csv", tp_results, delimiter=',')
np.savetxt("abs_diff_func5.csv", abs_diff, delimiter=',')

plt.plot(np.arange(0.05, 1.001, 0.001), (abs(maximin_results - tp_results)))
# plt.axvline(x=0.227597)
# plt.axvline(x=0.366691)
# plt.axvline(x=0.680123)
plt.xlabel("mu")
plt.ylabel("Absolute difference between maximin and threat point")
plt.title("Profit function threat point determination")
plt.savefig('plot6.png')