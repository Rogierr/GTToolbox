import numpy as np
import matplotlib.pyplot as plt
from computation.random_strategy_draw import random_strategy_draw

points = 100000

# initialize the payoff matrices
p1_payoff_PD = np.matrix('6 3; 8 4')
p2_payoff_PD = np.matrix('6 8; 3 4')

p1_payoff_BS = np.matrix('8 3; 6 4')
p2_payoff_BS = np.matrix('8 6; 3 4')

# draw some frequencies from which the rewards are drawn
play_frequencies = random_strategy_draw(points, p1_payoff_PD.size)

# compute the rewards for both players
rewards_p1_PD = np.sum(np.multiply(p1_payoff_PD.A1, play_frequencies), axis=1)
rewards_p2_PD = np.sum(np.multiply(p2_payoff_PD.A1, play_frequencies), axis=1)

rewards_p1_BS = np.sum(np.multiply(p1_payoff_BS.A1, play_frequencies), axis=1)
rewards_p2_BS = np.sum(np.multiply(p2_payoff_BS.A1, play_frequencies), axis=1)

# plot the simple repeated game rewards
plt.scatter(rewards_p1_PD, rewards_p2_PD, s=0.1)
plt.title("Prisoners Dilemma Data Sharing Rewards: Simple Repeated Game")
plt.show()

plt.scatter(rewards_p1_BS, rewards_p2_BS, s=0.1)
plt.title("Battle of the Sexes Data Sharing Rewards: Simple Repeated Game")
plt.show()

# apply the frequency vector functions
mu_p1_MB = play_frequencies[:,0] + float(2/3)*play_frequencies[:,1] + float(1/3)*play_frequencies[:,2]
mu_p2_MB = play_frequencies[:,0] + float(1/3)*play_frequencies[:,1] + float(2/3)*play_frequencies[:,2]

mu_p1_SB = float(2/3)*play_frequencies[:,0] + play_frequencies[:,1] + float(1/3)*play_frequencies[:,2]
mu_p2_SB = float(2/3)*play_frequencies[:,0] + float(1/3)*play_frequencies[:,1] + play_frequencies[:,2]

# define one of the two learning curve functions
def scurve(x):
    numerator = np.power(x, 3)
    denominator = 3*np.power(x, 2) - 3*x + 1

    return numerator/denominator

# compute the results of the multiple frequency vectors and learning curves
tanh_mu_p1_MB = 1+np.tanh((3*mu_p1_MB))
tanh_mu_p2_MB = 1+np.tanh((3*mu_p2_MB))

tanh_mu_p1_SB = 1+np.tanh((3*mu_p1_SB))
tanh_mu_p2_SB = 1+np.tanh((3*mu_p2_SB))

scurve_mu_p1_MB = 1+scurve(mu_p1_MB)
scurve_mu_p2_MB = 1+scurve(mu_p2_MB)

scurve_mu_p1_SB = 1+scurve(mu_p1_SB)
scurve_mu_p2_SB = 1+scurve(mu_p2_SB)

#adjusted payoff matrices are now computed
p1_rewards_BS_MB_tanh = np.multiply(tanh_mu_p1_MB, rewards_p1_BS)
p2_rewards_BS_MB_tanh = np.multiply(tanh_mu_p2_MB, rewards_p2_BS)

p1_rewards_BS_SB_tanh = np.multiply(tanh_mu_p1_SB, rewards_p1_BS)
p2_rewards_BS_SB_tanh = np.multiply(tanh_mu_p2_SB, rewards_p2_BS)

p1_rewards_BS_MB_curves = np.multiply(scurve_mu_p1_MB, rewards_p1_BS)
p2_rewards_BS_MB_curves = np.multiply(scurve_mu_p2_MB, rewards_p2_BS)

p1_rewards_BS_SB_curves = np.multiply(scurve_mu_p1_SB, rewards_p1_BS)
p2_rewards_BS_SB_curves = np.multiply(scurve_mu_p2_SB, rewards_p2_BS)


p1_rewards_PD_MB_tanh = np.multiply(tanh_mu_p1_MB, rewards_p1_PD)
p2_rewards_PD_MB_tanh = np.multiply(tanh_mu_p2_MB, rewards_p2_PD)

p1_rewards_PD_SB_tanh = np.multiply(tanh_mu_p1_SB, rewards_p1_PD)
p2_rewards_PD_SB_tanh = np.multiply(tanh_mu_p2_SB, rewards_p2_PD)

p1_rewards_PD_MB_curves = np.multiply(scurve_mu_p1_MB, rewards_p1_PD)
p2_rewards_PD_MB_curves = np.multiply(scurve_mu_p2_MB, rewards_p2_PD)

p1_rewards_PD_SB_curves = np.multiply(scurve_mu_p1_SB, rewards_p1_PD)
p2_rewards_PD_SB_curves = np.multiply(scurve_mu_p2_SB, rewards_p2_PD)

# now print all these beautiful results
# plt.scatter(p1_rewards_BS_MB_tanh, p2_rewards_BS_MB_tanh, s=0.1)
# plt.title("BS Data Sharing Rewards: Mutual Benefits with tanh")
# plt.show()
#
# plt.scatter(p1_rewards_BS_SB_tanh, p2_rewards_BS_SB_tanh, s=0.1)
# plt.title("BS Data Sharing Rewards: Singular Benefits with tanh")
# plt.show()
#
# plt.scatter(p1_rewards_BS_MB_curves, p2_rewards_BS_MB_curves, s=0.1)
# plt.title("BS Data Sharing Rewards: Mutual Benefits with scurve")
# plt.show()
#
# plt.scatter(p1_rewards_BS_SB_curves, p2_rewards_BS_SB_curves, s=0.1)
# plt.title("BS Data Sharing Rewards: Singular Benefits with scurve")
# plt.show()
#
#
# plt.scatter(p1_rewards_PD_MB_tanh, p2_rewards_PD_MB_tanh, s=0.1)
# plt.title("PD Data Sharing Rewards: Mutual Benefits with tanh")
# plt.show()
#
# plt.scatter(p1_rewards_PD_SB_tanh, p2_rewards_PD_SB_tanh, s=0.1)
# plt.title("PD Data Sharing Rewards: Singular Benefits with tanh")
# plt.show()
#
# plt.scatter(p1_rewards_PD_MB_curves, p2_rewards_PD_MB_curves, s=0.1)
# plt.title("PD Data Sharing Rewards: Mutual Benefits with scurve")
# plt.show()
#
# plt.scatter(p1_rewards_PD_SB_curves, p2_rewards_PD_SB_curves, s=0.1)
# plt.title("PD Data Sharing Rewards: Singular Benefits with scurve")
# plt.show()

## BELOW THIS LINE WE FOCUS ON THREAT POINT COMPUTATIONS

points_threat = 200000
under_threat_strategy = random_strategy_draw(points_threat, 2)

print(under_threat_strategy)

## HERE BELOW WE CAN COMPUTE THE PARETO EFFICIENT LINE

# print(rewards_p1_PD)
# print(rewards_p2_PD)
#
# print(np.sum([rewards_p1_PD, rewards_p2_PD], axis=0))