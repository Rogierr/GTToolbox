import numpy as np
from sympy import *

# COALITION VECTOR IS FILLED IN AS:
# 1, 2, 3, (1,2), (1,3), (2,3), (1,2,3)

coalition_vector = np.array([0, 0, 0, 4, 5, 4, 8])

print("Input coalition vector:")
print(coalition_vector)

print("")

print("Core computer")
x_1_max = (coalition_vector[6] - coalition_vector[5])
x_2_max = (coalition_vector[6] - coalition_vector[4])
x_3_max = (coalition_vector[6] - coalition_vector[3])
print("x_1 lower or equal to:", x_1_max)
print("x_2 lower or equal to:", x_2_max)
print("x_3 lower or equal to:", x_3_max)

print("Checking if the core is possibly non-empty:")
added_maxes = x_1_max + x_2_max + x_3_max
boundary_max = coalition_vector[6]
print(added_maxes >= boundary_max)
print("")


# SHAPLEY TABLE COMPUTATION IS FILLED IN WITH COALITIONS
# (1,2,3)
# (1,3,2)
# (2,1,3)
# (2,3,1)
# (3,1,2)
# (3,2,1)

shapley_table_computation = np.zeros((6, 3))

shapley_table_computation[0,0] = coalition_vector[0] - 0
shapley_table_computation[0,1] = coalition_vector[3] - coalition_vector[0]
shapley_table_computation[0,2] = coalition_vector[6] - coalition_vector[3]

shapley_table_computation[1,0] = coalition_vector[0] - 0
shapley_table_computation[1,2] = coalition_vector[4] - coalition_vector[0]
shapley_table_computation[1,1] = coalition_vector[6] - coalition_vector[4]

shapley_table_computation[2,1] = coalition_vector[1] - 0
shapley_table_computation[2,0] = coalition_vector[3] - coalition_vector[1]
shapley_table_computation[2,2] = coalition_vector[6] - coalition_vector[3]

shapley_table_computation[3,1] = coalition_vector[1] - 0
shapley_table_computation[3,2] = coalition_vector[5] - coalition_vector[1]
shapley_table_computation[3,0] = coalition_vector[6] - coalition_vector[5]

shapley_table_computation[4,2] = coalition_vector[2] - 0
shapley_table_computation[4,0] = coalition_vector[4] - coalition_vector[2]
shapley_table_computation[4,1] = coalition_vector[6] - coalition_vector[4]

shapley_table_computation[5,2] = coalition_vector[2] - 0
shapley_table_computation[5,1] = coalition_vector[5] - coalition_vector[2]
shapley_table_computation[5,0] = coalition_vector[6] - coalition_vector[5]

print("Resulting computations of the shapley table:")
print(shapley_table_computation)

print("")

summation_shapleys = np.zeros(3)
summation_shapleys[0] = np.sum(shapley_table_computation[:,0])
summation_shapleys[1] = np.sum(shapley_table_computation[:,1])
summation_shapleys[2] = np.sum(shapley_table_computation[:,2])

print("Summation of the shapley values")
print(summation_shapleys)

print("Shapley Value for Player 1: ", summation_shapleys[0],"/6")
print("Shapley Value for Player 2: ", summation_shapleys[1],"/6")
print("Shapley Value for Player 3: ", summation_shapleys[2],"/6")