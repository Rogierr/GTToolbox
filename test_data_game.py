import numpy as np
import matplotlib.pyplot as plt
from computation.random_strategy_draw import random_strategy_draw

points = 100000
phi = 0

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
# plt.scatter(rewards_p1_PD, rewards_p2_PD, s=0.1)
# plt.title("Prisoners Dilemma Data Sharing Rewards: Simple Repeated Game")
# plt.show()
#
# plt.scatter(rewards_p1_BS, rewards_p2_BS, s=0.1)
# plt.title("Battle of the Sexes Data Sharing Rewards: Simple Repeated Game")
# plt.show()

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
tanh_mu_p1_MB = phi+np.tanh((3*mu_p1_MB))
tanh_mu_p2_MB = phi+np.tanh((3*mu_p2_MB))

tanh_mu_p1_SB = phi+np.tanh((3*mu_p1_SB))
tanh_mu_p2_SB = phi+np.tanh((3*mu_p2_SB))

scurve_mu_p1_MB = phi+scurve(mu_p1_MB)
scurve_mu_p2_MB = phi+scurve(mu_p2_MB)

scurve_mu_p1_SB = phi+scurve(mu_p1_SB)
scurve_mu_p2_SB = phi+scurve(mu_p2_SB)

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

## BELOW THIS LINE WE FOCUS ON THREAT POINT COMPUTATIONS

# draw some random strategies
points_threat = 200000
threat_strategy = random_strategy_draw(points_threat, 2)

# convert these strategies into best reply matrices
p1_br_up_MB = threat_strategy[:, 0] + float(2 / 3) * threat_strategy[:, 1]
p1_br_down_MB = float(1/3) * threat_strategy[:, 0]

p2_br_left_MB = threat_strategy[:, 0] + float(2 / 3) * threat_strategy[:, 1]
p2_br_right_MB = float(1/3) * threat_strategy[:, 0]

p1_br_up_SB = float(2/3) * threat_strategy[:, 0] + threat_strategy[:, 1]
p1_br_down_SB = float(1/3) * threat_strategy[:, 0]

p2_br_left_SB = float(2/3) * threat_strategy[:, 0] + threat_strategy[:, 1]
p2_br_right_SB = float(1/3) * threat_strategy[:, 0]

# compute the results of the FD function
tanh_p1_br_up_MB = phi+np.tanh(3*p1_br_up_MB)
tanh_p1_br_down_MB = phi+np.tanh(3*p1_br_down_MB)

tanh_p2_br_left_MB = phi+np.tanh(3*p2_br_left_MB)
tanh_p2_br_right_MB = phi+np.tanh(3*p2_br_right_MB)

tanh_p1_br_up_SB = phi+np.tanh(3*p1_br_up_SB)
tanh_p1_br_down_SB = phi+np.tanh(3*p1_br_down_SB)

tanh_p2_br_left_SB = phi+np.tanh(3*p2_br_left_SB)
tanh_p2_br_right_SB = phi+np.tanh(3*p2_br_right_SB)

scurve_p1_br_up_MB = phi+scurve(p1_br_up_MB)
scurve_p1_br_down_MB = phi+scurve(p1_br_down_MB)

scurve_p2_br_left_MB = phi+scurve(p2_br_left_MB)
scurve_p2_br_right_MB = phi+scurve(p2_br_right_MB)

scurve_p1_br_up_SB = phi+scurve(p1_br_up_SB)
scurve_p1_br_down_SB = phi+scurve(p1_br_down_SB)

scurve_p2_br_left_SB = phi+scurve(p2_br_left_SB)
scurve_p2_br_right_SB = phi+scurve(p2_br_right_SB)

# compute the rewards based on the chosen threat point strategies,
rewards_p1_up_PD = np.dot(threat_strategy, p1_payoff_PD.A1[0:2])
rewards_p1_down_PD = np.dot(threat_strategy, p1_payoff_PD.A1[2:4])

rewards_p2_left_PD = np.dot(threat_strategy, [p2_payoff_PD.A1[0], p2_payoff_PD.A1[2]])
rewards_p2_right_PD = np.dot(threat_strategy, [p2_payoff_PD.A1[1], p2_payoff_PD.A1[3]])

rewards_p1_up_BS = np.dot(threat_strategy, p1_payoff_BS.A1[0:2])
rewards_p1_down_BS = np.dot(threat_strategy, p1_payoff_BS.A1[2:4])

rewards_p2_left_BS = np.dot(threat_strategy, [p2_payoff_BS.A1[0], p2_payoff_BS.A1[2]])
rewards_p2_right_BS = np.dot(threat_strategy, [p2_payoff_BS.A1[1], p2_payoff_BS.A1[3]])

# compute the FD adjusted rewards for threat point determination
rewards_p1_up_PD_tanh_MB = np.multiply(tanh_p1_br_up_MB, rewards_p1_up_PD)
rewards_p1_down_PD_tanh_MB = np.multiply(tanh_p1_br_down_MB, rewards_p1_down_PD)

rewards_p2_left_PD_tanh_MB = np.multiply(tanh_p2_br_left_MB, rewards_p2_left_PD)
rewards_p2_right_PD_tanh_MB = np.multiply(tanh_p2_br_right_MB, rewards_p2_right_PD)

rewards_p1_up_PD_scurve_MB = np.multiply(scurve_p1_br_up_MB, rewards_p1_up_PD)
rewards_p1_down_PD_scurve_MB = np.multiply(scurve_p1_br_down_MB, rewards_p1_down_PD)

rewards_p2_left_PD_scurve_MB = np.multiply(scurve_p2_br_left_MB, rewards_p2_left_PD)
rewards_p2_right_PD_scurve_MB = np.multiply(scurve_p2_br_right_MB, rewards_p2_right_PD)

rewards_p1_up_PD_tanh_SB = np.multiply(tanh_p1_br_up_SB, rewards_p1_up_PD)
rewards_p1_down_PD_tanh_SB = np.multiply(tanh_p1_br_down_SB, rewards_p1_down_PD)

rewards_p2_left_PD_tanh_SB = np.multiply(tanh_p2_br_left_SB, rewards_p2_left_PD)
rewards_p2_right_PD_tanh_SB = np.multiply(tanh_p2_br_right_SB, rewards_p2_right_PD)

rewards_p1_up_PD_scurve_SB = np.multiply(scurve_p1_br_up_SB, rewards_p1_up_PD)
rewards_p1_down_PD_scurve_SB = np.multiply(scurve_p1_br_down_SB, rewards_p1_down_PD)

rewards_p2_left_PD_scurve_SB = np.multiply(scurve_p2_br_left_SB, rewards_p2_left_PD)
rewards_p2_right_PD_scurve_SB = np.multiply(scurve_p2_br_right_SB, rewards_p2_right_PD)


rewards_p1_up_BS_tanh_MB = np.multiply(tanh_p1_br_up_MB, rewards_p1_up_BS)
rewards_p1_down_BS_tanh_MB = np.multiply(tanh_p1_br_down_MB, rewards_p1_down_BS)

rewards_p2_left_BS_tanh_MB = np.multiply(tanh_p2_br_left_MB, rewards_p2_left_BS)
rewards_p2_right_BS_tanh_MB = np.multiply(tanh_p2_br_right_MB, rewards_p2_right_BS)

rewards_p1_up_BS_scurve_MB = np.multiply(scurve_p1_br_up_MB, rewards_p1_up_BS)
rewards_p1_down_BS_scurve_MB = np.multiply(scurve_p1_br_down_MB, rewards_p1_down_BS)

rewards_p2_left_BS_scurve_MB = np.multiply(scurve_p2_br_left_MB, rewards_p2_left_BS)
rewards_p2_right_BS_scurve_MB = np.multiply(scurve_p2_br_right_MB, rewards_p2_right_BS)

rewards_p1_up_BS_tanh_SB = np.multiply(tanh_p1_br_up_SB, rewards_p1_up_BS)
rewards_p1_down_BS_tanh_SB = np.multiply(tanh_p1_br_down_SB, rewards_p1_down_BS)

rewards_p2_left_BS_tanh_SB = np.multiply(tanh_p2_br_left_SB, rewards_p2_left_BS)
rewards_p2_right_BS_tanh_SB = np.multiply(tanh_p2_br_right_SB, rewards_p2_right_BS)

rewards_p1_up_BS_scurve_SB = np.multiply(scurve_p1_br_up_SB, rewards_p1_up_BS)
rewards_p1_down_BS_scurve_SB = np.multiply(scurve_p1_br_down_SB, rewards_p1_down_BS)

rewards_p2_left_BS_scurve_SB = np.multiply(scurve_p2_br_left_SB, rewards_p2_left_BS)
rewards_p2_right_BS_scurve_SB = np.multiply(scurve_p2_br_right_SB, rewards_p2_right_BS)


# cluster the threat point but first initialize

threat_point_p1_PD_tanh_MB = np.zeros((points_threat, 2))
threat_point_p2_PD_tanh_MB = np.zeros((points_threat, 2))

threat_point_p1_PD_scurve_MB = np.zeros((points_threat, 2))
threat_point_p2_PD_scurve_MB = np.zeros((points_threat, 2))

threat_point_p1_PD_tanh_SB = np.zeros((points_threat, 2))
threat_point_p2_PD_tanh_SB = np.zeros((points_threat, 2))

threat_point_p1_PD_scurve_SB = np.zeros((points_threat, 2))
threat_point_p2_PD_scurve_SB = np.zeros((points_threat, 2))

threat_point_p1_BS_tanh_MB = np.zeros((points_threat, 2))
threat_point_p2_BS_tanh_MB = np.zeros((points_threat, 2))

threat_point_p1_BS_scurve_MB = np.zeros((points_threat, 2))
threat_point_p2_BS_scurve_MB = np.zeros((points_threat, 2))

threat_point_p1_BS_tanh_SB = np.zeros((points_threat, 2))
threat_point_p2_BS_tanh_SB = np.zeros((points_threat, 2))

threat_point_p1_BS_scurve_SB = np.zeros((points_threat, 2))
threat_point_p2_BS_scurve_SB = np.zeros((points_threat, 2))


## UNTIL HERE IT IS DONE AND CHECKED
# fill in the threat point matrices with the rewards

threat_point_p1_PD_tanh_MB[:, 0] = rewards_p1_up_PD_tanh_MB
threat_point_p1_PD_tanh_MB[:, 1] = rewards_p1_down_PD_tanh_MB

threat_point_p2_PD_tanh_MB[:, 0] = rewards_p2_left_PD_tanh_MB
threat_point_p2_PD_tanh_MB[:, 1] = rewards_p2_right_PD_tanh_MB

threat_point_p1_PD_scurve_MB[:, 0] = rewards_p1_up_PD_scurve_MB
threat_point_p1_PD_scurve_MB[:, 1] = rewards_p1_down_PD_scurve_MB

threat_point_p2_PD_scurve_MB[:, 0] = rewards_p2_left_PD_scurve_MB
threat_point_p2_PD_scurve_MB[:, 1] = rewards_p2_right_PD_scurve_MB

threat_point_p1_PD_tanh_SB[:, 0] = rewards_p1_up_PD_tanh_SB
threat_point_p1_PD_tanh_SB[:, 1] = rewards_p1_down_PD_tanh_SB

threat_point_p2_PD_tanh_SB[:, 0] = rewards_p2_left_PD_tanh_SB
threat_point_p2_PD_tanh_SB[:, 1] = rewards_p2_right_PD_tanh_SB

threat_point_p1_PD_scurve_SB[:, 0] = rewards_p1_up_PD_scurve_SB
threat_point_p1_PD_scurve_SB[:, 1] = rewards_p1_down_PD_scurve_SB

threat_point_p2_PD_scurve_SB[:, 0] = rewards_p2_left_PD_scurve_SB
threat_point_p2_PD_scurve_SB[:, 1] = rewards_p2_right_PD_scurve_SB


threat_point_p1_BS_tanh_MB[:, 0] = rewards_p1_up_BS_tanh_MB
threat_point_p1_BS_tanh_MB[:, 1] = rewards_p1_down_BS_tanh_MB

threat_point_p2_BS_tanh_MB[:, 0] = rewards_p2_left_BS_tanh_MB
threat_point_p2_BS_tanh_MB[:, 1] = rewards_p2_right_BS_tanh_MB

threat_point_p1_BS_scurve_MB[:, 0] = rewards_p1_up_BS_scurve_MB
threat_point_p1_BS_scurve_MB[:, 1] = rewards_p1_down_BS_scurve_MB

threat_point_p2_BS_scurve_MB[:, 0] = rewards_p2_left_BS_scurve_MB
threat_point_p2_BS_scurve_MB[:, 1] = rewards_p2_right_BS_scurve_MB

threat_point_p1_BS_tanh_SB[:, 0] = rewards_p1_up_BS_tanh_SB
threat_point_p1_BS_tanh_SB[:, 1] = rewards_p1_down_BS_tanh_SB

threat_point_p2_BS_tanh_SB[:, 0] = rewards_p2_left_BS_tanh_SB
threat_point_p2_BS_tanh_SB[:, 1] = rewards_p2_right_BS_tanh_SB

threat_point_p1_BS_scurve_SB[:, 0] = rewards_p1_up_BS_scurve_SB
threat_point_p1_BS_scurve_SB[:, 1] = rewards_p1_down_BS_scurve_SB

threat_point_p2_BS_scurve_SB[:, 0] = rewards_p2_left_BS_scurve_SB
threat_point_p2_BS_scurve_SB[:, 1] = rewards_p2_right_BS_scurve_SB

# determine the threat point
TPResult_p1_PD_tanh_MB = np.min(np.max(threat_point_p1_PD_tanh_MB, axis=1), axis=0)
TPResult_p2_PD_tanh_MB = np.min(np.max(threat_point_p2_PD_tanh_MB, axis=1), axis=0)

TPResult_p1_PD_scurve_MB = np.min(np.max(threat_point_p1_PD_scurve_MB, axis=1), axis=0)
TPResult_p2_PD_scurve_MB = np.min(np.max(threat_point_p2_PD_scurve_MB, axis=1), axis=0)

TPResult_p1_PD_tanh_SB = np.min(np.max(threat_point_p1_PD_tanh_SB, axis=1), axis=0)
TPResult_p2_PD_tanh_SB = np.min(np.max(threat_point_p2_PD_tanh_SB, axis=1), axis=0)

TPResult_p1_PD_scurve_SB = np.min(np.max(threat_point_p1_PD_scurve_MB, axis=1), axis=0)
TPResult_p2_PD_scurve_SB = np.min(np.max(threat_point_p2_PD_scurve_MB, axis=1), axis=0)

TPResult_p1_BS_tanh_MB = np.min(np.max(threat_point_p1_BS_tanh_MB, axis=1), axis=0)
TPResult_p2_BS_tanh_MB = np.min(np.max(threat_point_p2_BS_tanh_MB, axis=1), axis=0)

TPResult_p1_BS_scurve_MB = np.min(np.max(threat_point_p1_BS_scurve_MB, axis=1), axis=0)
TPResult_p2_BS_scurve_MB = np.min(np.max(threat_point_p2_BS_scurve_MB, axis=1), axis=0)

TPResult_p1_BS_tanh_SB = np.min(np.max(threat_point_p1_BS_tanh_SB, axis=1), axis=0)
TPResult_p2_BS_tanh_SB = np.min(np.max(threat_point_p2_BS_tanh_SB, axis=1), axis=0)

TPResult_p1_BS_scurve_SB = np.min(np.max(threat_point_p1_BS_scurve_MB, axis=1), axis=0)
TPResult_p2_BS_scurve_SB = np.min(np.max(threat_point_p2_BS_scurve_MB, axis=1), axis=0)



plt.scatter(p1_rewards_PD_MB_tanh, p2_rewards_PD_MB_tanh, s=0.1, color='C0', label='Rewards')
plt.title("PD Data Sharing Rewards: Mutual Benefits with tanh")
plt.scatter(TPResult_p1_PD_tanh_MB, TPResult_p2_PD_tanh_MB, color='r', label='Threat point')
print("Threat point reward found for PD tanh with MB at:", [TPResult_p1_PD_tanh_MB, TPResult_p2_PD_tanh_MB])
plt.show()

plt.scatter(p1_rewards_PD_SB_tanh, p2_rewards_PD_SB_tanh, s=0.1, color='C0', label='Rewards')
plt.title("PD Data Sharing Rewards: Singular Benefits with tanh")
plt.scatter(TPResult_p1_PD_tanh_SB, TPResult_p2_PD_tanh_SB, color='r', label='Threat point')
print("Threat point reward found for PD tanh with SB at:", [TPResult_p1_PD_tanh_SB, TPResult_p2_PD_tanh_SB])
plt.show()

plt.scatter(p1_rewards_PD_MB_curves, p2_rewards_PD_MB_curves, s=0.1, label='Rewards')
plt.title("PD Data Sharing Rewards: Mutual Benefits with scurve")
plt.scatter(TPResult_p1_PD_scurve_MB, TPResult_p2_PD_scurve_MB, color='r', label='Threat point')
print("Threat point reward found for PD scurve with MB at:", [TPResult_p1_PD_scurve_MB, TPResult_p2_PD_scurve_MB])
plt.show()

plt.scatter(p1_rewards_PD_SB_curves, p2_rewards_PD_SB_curves, s=0.1, label='Rewards')
plt.title("PD Data Sharing Rewards: Singular Benefits with scurve")
plt.scatter(TPResult_p1_PD_scurve_SB, TPResult_p2_PD_scurve_SB, color='r', label='Threat point')
print("Threat point reward found for PD scurve with SB at:", [TPResult_p1_PD_scurve_SB, TPResult_p2_PD_scurve_SB])
plt.show()


plt.scatter(p1_rewards_BS_MB_tanh, p2_rewards_BS_MB_tanh, s=0.1, color='C0', label='Rewards')
plt.title("BS Data Sharing Rewards: Mutual Benefits with tanh")
plt.scatter(TPResult_p1_BS_tanh_MB, TPResult_p2_BS_tanh_MB, color='r', label='Threat point')
print("Threat point reward found for BS tanh with MB at:", [TPResult_p1_BS_tanh_MB, TPResult_p2_BS_tanh_MB])
plt.show()

plt.scatter(p1_rewards_BS_SB_tanh, p2_rewards_BS_SB_tanh, s=0.1, color='C0', label='Rewards')
plt.title("BS Data Sharing Rewards: Singular Benefits with tanh")
plt.scatter(TPResult_p1_BS_tanh_SB, TPResult_p2_BS_tanh_SB, color='r', label='Threat point')
print("Threat point reward found for BS tanh with SB at:", [TPResult_p1_BS_tanh_SB, TPResult_p2_BS_tanh_SB])
plt.show()

plt.scatter(p1_rewards_BS_MB_curves, p2_rewards_BS_MB_curves, s=0.1, label='Rewards')
plt.title("BS Data Sharing Rewards: Mutual Benefits with scurve")
plt.scatter(TPResult_p1_BS_scurve_MB, TPResult_p2_BS_scurve_MB, color='r', label='Threat point')
print("Threat point reward found for BS scurve with MB at:", [TPResult_p1_BS_scurve_MB, TPResult_p2_BS_scurve_MB])
plt.show()

plt.scatter(p1_rewards_BS_SB_curves, p2_rewards_BS_SB_curves, s=0.1, label='Rewards')
plt.title("BS Data Sharing Rewards: Singular Benefits with scurve")
plt.scatter(TPResult_p1_BS_scurve_SB, TPResult_p2_BS_scurve_SB, color='r', label='Threat point')
print("Threat point reward found for BS scurve with SB at:", [TPResult_p1_BS_scurve_SB, TPResult_p2_BS_scurve_SB])
plt.show()


## HERE BELOW WE CAN COMPUTE THE PARETO EFFICIENT LINE

# print(rewards_p1_PD)
# print(rewards_p2_PD)
#
# print(np.sum([rewards_p1_PD, rewards_p2_PD], axis=0))