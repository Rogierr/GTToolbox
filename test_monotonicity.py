import numpy as np
import time

from computation.random_strategy_draw import random_strategy_draw
from FD_functions.rho_function import rho_function
from FD_functions.mu_function import mu_function
from FD_functions.profit_function import profit_function
from game.game import TypeICTP

p1 = np.matrix('16 14; 28 26')
p2 = np.matrix('16 28; 14 26')

testgame = TypeICTP(p1, p2)

points = 10

start_time = time.time()

draw_payoffs = random_strategy_draw(points, 2)
print(draw_payoffs)

p1_actions_range = np.arange(2)
p2_actions_range = np.arange(2)

frequency_pairs = np.zeros((points*2, 2*2))

for i in np.nditer(p1_actions_range):
    for j in np.nditer(p2_actions_range):
        frequency_pairs[i*points:points*(i+1),(2*i)+j] = draw_payoffs[:,j]

rho_matrix = rho_function(frequency_pairs)

print("The result of the rho matrix")
print(rho_matrix)

mu_matrix = mu_function(testgame, rho_matrix)
print("The result of the mu matrix")
print(mu_matrix)

profit_matrix = profit_function(mu_matrix)
print("The result of the profit matrix")
print(profit_matrix)

monoto_matrix = np.zeros(points*2)
monoto_matrix[mu_matrix > 0.366691233367131] = 1

print(monoto_matrix)

result_game_p1 = np.dot(frequency_pairs, p1.A1)

result_game_p1_rarity = np.multiply(result_game_p1, profit_matrix)

print("The results of the game for player 1")
print(result_game_p1_rarity)

payoffs_sort = np.zeros((points, 2))
points_range = np.arange(10)
actions_range = np.arange(2)


for x in np.nditer(points_range):
    for i in np.nditer(actions_range):
        payoffs_sort[x,i] = result_game_p1_rarity[points*i+x]

print("")
print("")
print("The sorted payoff matrix, threat point determination should take place on this")
print(payoffs_sort)

print("")
print("")
print("Determining the threat point for p1")
print(np.min(np.max(payoffs_sort, axis=1)))

end_time = time.time()

total_time = end_time - start_time
print("")
print("In a time of")
print(total_time, "seconds")
