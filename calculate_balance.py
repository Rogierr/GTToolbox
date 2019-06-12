import numpy as np
from game.game import ETPGame
from FD_functions.rho_function import rho_function
from FD_functions.mu_function import mu_function
from FD_functions.profit_function import profit_function
from computation.balance_equation_all import balance_equation_all

p1_1 = np.matrix('16 14; 28 26')
p2_1 = np.matrix('16 28; 14 26')

p1_2 = np.matrix('4 3.5; 7 6.5')
p2_2 = np.matrix('4 7; 3.5 6.5')

ts1_1 = np.matrix('16 14; 28 24')
ts2_1 = np.matrix('16 28; 14 24')

ts1_2 = np.matrix('4 3.5; 7 6')
ts2_2 = np.matrix('4 7; 3.5 6')

trans1_1 = np.matrix('0.8 0.7; 0.7 0.6')
trans2_1 = np.matrix('0.5 0.4; 0.4 0.15')

trans1_2 = np.matrix('0.2 0.3; 0.3 0.4')
trans2_2 = np.matrix('0.5 0.6; 0.6 0.85')

matrixA = np.matrix('0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.35 0.3 0.3 0.25 0.2 0.15 0.15 0.05; 0.35 0.3 0.3 0.25 0.2 0.15 0.15 0.05; 0.7 0.6 0.6 0.5 0.4 0.3 0.3 0.1; 0 0 0 0 0 0 0 0; 0.35 0.3 0.3 0.25 0.2 0.15 0.15 0.05; 0.35 0.3 0.3 0.25 0.2 0.15 0.15 0.05; 0.7 0.6 0.6 0.5 0.4 0.3 0.3 0.1')
matrixB = np.matrix('0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.0 0.0 0.0 0.00 0.0 0.0 0.0 0.00; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0 0 0 0 0 0 0 0; 0.0 0.0 0.0 0.00 0.0 0.00 0.00 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.00; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0')
matrixC = np.matrix('0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.57 0.5 0.5 0.45 0.39 0.315 0.315 0.12; 0.00 0.0 0.0 0.00 0.0 0.00 0.00 0.00; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.285 0.25 0.25 0.225 0.195 0.1575 0.1575 0.06; 0.57 0.5 0.5 0.45 0.39 0.315 0.315 0.12')

ETP = ETPGame(p1_1,p2_1,p1_2,p2_2,trans1_1,trans2_1,trans1_2,trans2_2,matrixC)
ETP.activate_rarity()
# ETP.plotting_rare("Rarity")
ETP.activate_hysteresis(1.5)
ETP.adjust_mu(1)


x = np.array([[5/7, 0, 0, 0, 2/7, 0, 0, 0], [0, 0, 0, 0.5, 0, 0, 0, 0.5]])

res = balance_equation_all(ETP, 2, x)
print(res)

sus = res[0]
nr = res[1]

fd = mu_function(ETP, rho_function(res))
print(fd)

profit = profit_function(fd)
print(profit)

p_sus = fd[0] * (sus[0]*16 + sus[4]*4)
result_sus = [p_sus, p_sus]
print("The result of x sus:", result_sus)

p_nr = fd[1] * (nr[3]*26 + nr[7]*6.5)
result_nr = [p_nr, p_nr]
print("The result of x nr:", result_nr)

p_rarity_sus = p_sus * profit[0]
p_rarity_nr = p_nr * profit[1]

print("The result of x sus with rarity:", [p_rarity_sus, p_rarity_sus])
print("The result of x nr with rarity:", [p_rarity_nr, p_rarity_nr])