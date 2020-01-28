import numpy as np
import matplotlib.pyplot as plt
from computation.random_strategy_draw import random_strategy_draw
from game.game import RepeatedGame
from FD_functions.learning_curves import scurve

points = 2500000

# initialize the payoff matrices
p1_payoff_PD = np.matrix('6 3; 8 4')
p2_payoff_PD = np.matrix('6 8; 3 4')

Prisoners_Dilemma = RepeatedGame(p1_payoff_PD, p2_payoff_PD)

p1_payoff_SH = np.matrix('8 3; 6 4')
p2_payoff_SH = np.matrix('8 6; 3 4')

Stag_Hunt = RepeatedGame(p1_payoff_SH, p2_payoff_SH)

# Prisoners_Dilemma.plot_all_rewards(points, "Prisoners Dilemma - Data Dilemma")
# Stag_Hunt.plot_all_rewards(points, "Stag Hunt - Data Dilemma")

for i in range(0, 1):
    Prisoners_Dilemma.set_phi(i*0.5+0)
    Prisoners_Dilemma.define_learning_curve('tanh')
    Prisoners_Dilemma.define_mu('sb')
    # Prisoners_Dilemma.plot_all_rewards(points, 'Tanh Prisoners Dilemma SB')
    print("Threat point PD tanh sb")
    Prisoners_Dilemma.compute_threat_point(100000, True, True, True)
    # Prisoners_Dilemma.compute_maximin(5, True, True)

    Prisoners_Dilemma.define_learning_curve('tanh')
    Prisoners_Dilemma.define_mu('mb')
    # Prisoners_Dilemma.plot_all_rewards(points, 'Tanh Prisoners Dilemma MB')
    print("Threat point PD tanh MB")
    Prisoners_Dilemma.compute_threat_point(100000, True, True, True)
    # Prisoners_Dilemma.compute_maximin(100000, True, True)

    Prisoners_Dilemma.define_learning_curve('scurve')
    Prisoners_Dilemma.define_mu('sb')
    # Prisoners_Dilemma.plot_all_rewards(points, 'Scurve Prisoners Dilemma SB')
    print("Threat point PD scurve SB")
    Prisoners_Dilemma.compute_threat_point(100000, True, True, True)

    Prisoners_Dilemma.define_learning_curve('scurve')
    Prisoners_Dilemma.define_mu('mb')
    # Prisoners_Dilemma.plot_all_rewards(points, 'Scurve Prisoners Dilemma MB')
    print("Threat point PD scurve MB")
    Prisoners_Dilemma.compute_threat_point(100000, True, True, True)


    Stag_Hunt.define_learning_curve('tanh')
    Stag_Hunt.define_mu('sb')
    # Stag_Hunt.plot_all_rewards(points, 'Tanh Stag Hunt SB')
    print("Threat point SH tanh SB")
    Stag_Hunt.compute_threat_point(100000, True, True, True)

    Stag_Hunt.define_learning_curve('tanh')
    Stag_Hunt.define_mu('mb')
    # Stag_Hunt.plot_all_rewards(points, 'Tanh Stag Hunt MB')
    print("Threat point SH tanh MB")
    Stag_Hunt.compute_threat_point(100000, True, True, True)

    Stag_Hunt.define_learning_curve('scurve')
    Stag_Hunt.define_mu('sb')
    # Stag_Hunt.plot_all_rewards(points, 'Scurve Stag Hunt SB')
    print("Threat point SH Scurve SB")
    Stag_Hunt.compute_threat_point(100000, True, True, True)

    Stag_Hunt.define_learning_curve('scurve')
    Stag_Hunt.define_mu('mb')
    # Stag_Hunt.plot_all_rewards(points, 'Scurve Stag Hunt MB')
    print("Threat point SH scurve MB")
    Stag_Hunt.compute_threat_point(100000, True, True, True)

## BELOW THIS LINE WE FOCUS ON THREAT POINT COMPUTATIONS

# draw some random strategies
points_threat = 200000
threat_strategy = random_strategy_draw(points_threat, 2)
phi = 0

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

rewards_p1_up_SH = np.dot(threat_strategy, p1_payoff_SH.A1[0:2])
rewards_p1_down_SH = np.dot(threat_strategy, p1_payoff_SH.A1[2:4])

rewards_p2_left_SH = np.dot(threat_strategy, [p2_payoff_SH.A1[0], p2_payoff_SH.A1[2]])
rewards_p2_right_SH = np.dot(threat_strategy, [p2_payoff_SH.A1[1], p2_payoff_SH.A1[3]])

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


rewards_p1_up_SH_tanh_MB = np.multiply(tanh_p1_br_up_MB, rewards_p1_up_SH)
rewards_p1_down_SH_tanh_MB = np.multiply(tanh_p1_br_down_MB, rewards_p1_down_SH)

rewards_p2_left_SH_tanh_MB = np.multiply(tanh_p2_br_left_MB, rewards_p2_left_SH)
rewards_p2_right_SH_tanh_MB = np.multiply(tanh_p2_br_right_MB, rewards_p2_right_SH)

rewards_p1_up_SH_scurve_MB = np.multiply(scurve_p1_br_up_MB, rewards_p1_up_SH)
rewards_p1_down_SH_scurve_MB = np.multiply(scurve_p1_br_down_MB, rewards_p1_down_SH)

rewards_p2_left_SH_scurve_MB = np.multiply(scurve_p2_br_left_MB, rewards_p2_left_SH)
rewards_p2_right_SH_scurve_MB = np.multiply(scurve_p2_br_right_MB, rewards_p2_right_SH)

rewards_p1_up_SH_tanh_SB = np.multiply(tanh_p1_br_up_SB, rewards_p1_up_SH)
rewards_p1_down_SH_tanh_SB = np.multiply(tanh_p1_br_down_SB, rewards_p1_down_SH)

rewards_p2_left_SH_tanh_SB = np.multiply(tanh_p2_br_left_SB, rewards_p2_left_SH)
rewards_p2_right_SH_tanh_SB = np.multiply(tanh_p2_br_right_SB, rewards_p2_right_SH)

rewards_p1_up_SH_scurve_SB = np.multiply(scurve_p1_br_up_SB, rewards_p1_up_SH)
rewards_p1_down_SH_scurve_SB = np.multiply(scurve_p1_br_down_SB, rewards_p1_down_SH)

rewards_p2_left_SH_scurve_SB = np.multiply(scurve_p2_br_left_SB, rewards_p2_left_SH)
rewards_p2_right_SH_scurve_SB = np.multiply(scurve_p2_br_right_SB, rewards_p2_right_SH)


# cluster the threat point but first initialize

threat_point_p1_PD_tanh_MB = np.zeros((points_threat, 2))
threat_point_p2_PD_tanh_MB = np.zeros((points_threat, 2))

threat_point_p1_PD_scurve_MB = np.zeros((points_threat, 2))
threat_point_p2_PD_scurve_MB = np.zeros((points_threat, 2))

threat_point_p1_PD_tanh_SB = np.zeros((points_threat, 2))
threat_point_p2_PD_tanh_SB = np.zeros((points_threat, 2))

threat_point_p1_PD_scurve_SB = np.zeros((points_threat, 2))
threat_point_p2_PD_scurve_SB = np.zeros((points_threat, 2))

threat_point_p1_SH_tanh_MB = np.zeros((points_threat, 2))
threat_point_p2_SH_tanh_MB = np.zeros((points_threat, 2))

threat_point_p1_SH_scurve_MB = np.zeros((points_threat, 2))
threat_point_p2_SH_scurve_MB = np.zeros((points_threat, 2))

threat_point_p1_SH_tanh_SB = np.zeros((points_threat, 2))
threat_point_p2_SH_tanh_SB = np.zeros((points_threat, 2))

threat_point_p1_SH_scurve_SB = np.zeros((points_threat, 2))
threat_point_p2_SH_scurve_SB = np.zeros((points_threat, 2))


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


threat_point_p1_SH_tanh_MB[:, 0] = rewards_p1_up_SH_tanh_MB
threat_point_p1_SH_tanh_MB[:, 1] = rewards_p1_down_SH_tanh_MB

threat_point_p2_SH_tanh_MB[:, 0] = rewards_p2_left_SH_tanh_MB
threat_point_p2_SH_tanh_MB[:, 1] = rewards_p2_right_SH_tanh_MB

threat_point_p1_SH_scurve_MB[:, 0] = rewards_p1_up_SH_scurve_MB
threat_point_p1_SH_scurve_MB[:, 1] = rewards_p1_down_SH_scurve_MB

threat_point_p2_SH_scurve_MB[:, 0] = rewards_p2_left_SH_scurve_MB
threat_point_p2_SH_scurve_MB[:, 1] = rewards_p2_right_SH_scurve_MB

threat_point_p1_SH_tanh_SB[:, 0] = rewards_p1_up_SH_tanh_SB
threat_point_p1_SH_tanh_SB[:, 1] = rewards_p1_down_SH_tanh_SB

threat_point_p2_SH_tanh_SB[:, 0] = rewards_p2_left_SH_tanh_SB
threat_point_p2_SH_tanh_SB[:, 1] = rewards_p2_right_SH_tanh_SB

threat_point_p1_SH_scurve_SB[:, 0] = rewards_p1_up_SH_scurve_SB
threat_point_p1_SH_scurve_SB[:, 1] = rewards_p1_down_SH_scurve_SB

threat_point_p2_SH_scurve_SB[:, 0] = rewards_p2_left_SH_scurve_SB
threat_point_p2_SH_scurve_SB[:, 1] = rewards_p2_right_SH_scurve_SB

# determine the threat point
TPResult_p1_PD_tanh_MB = np.min(np.max(threat_point_p1_PD_tanh_MB, axis=1), axis=0)
TPResult_p2_PD_tanh_MB = np.min(np.max(threat_point_p2_PD_tanh_MB, axis=1), axis=0)

TPResult_p1_PD_scurve_MB = np.min(np.max(threat_point_p1_PD_scurve_MB, axis=1), axis=0)
TPResult_p2_PD_scurve_MB = np.min(np.max(threat_point_p2_PD_scurve_MB, axis=1), axis=0)

TPResult_p1_PD_tanh_SB = np.min(np.max(threat_point_p1_PD_tanh_SB, axis=1), axis=0)
TPResult_p2_PD_tanh_SB = np.min(np.max(threat_point_p2_PD_tanh_SB, axis=1), axis=0)

TPResult_p1_PD_scurve_SB = np.min(np.max(threat_point_p1_PD_scurve_SB, axis=1), axis=0)
TPResult_p2_PD_scurve_SB = np.min(np.max(threat_point_p2_PD_scurve_SB, axis=1), axis=0)

TPResult_p1_SH_tanh_MB = np.min(np.max(threat_point_p1_SH_tanh_MB, axis=1), axis=0)
TPResult_p2_SH_tanh_MB = np.min(np.max(threat_point_p2_SH_tanh_MB, axis=1), axis=0)

TPResult_p1_SH_scurve_MB = np.min(np.max(threat_point_p1_SH_scurve_MB, axis=1), axis=0)
TPResult_p2_SH_scurve_MB = np.min(np.max(threat_point_p2_SH_scurve_MB, axis=1), axis=0)

TPResult_p1_SH_tanh_SB = np.min(np.max(threat_point_p1_SH_tanh_SB, axis=1), axis=0)
TPResult_p2_SH_tanh_SB = np.min(np.max(threat_point_p2_SH_tanh_SB, axis=1), axis=0)

TPResult_p1_SH_scurve_SB = np.min(np.max(threat_point_p1_SH_scurve_SB, axis=1), axis=0)
TPResult_p2_SH_scurve_SB = np.min(np.max(threat_point_p2_SH_scurve_SB, axis=1), axis=0)

print("PD Tanh SB", [TPResult_p1_PD_tanh_SB, TPResult_p2_PD_tanh_SB])
print("PD tanh MB", [TPResult_p1_PD_tanh_MB, TPResult_p2_PD_tanh_MB])
print("PD Scurve SB", [TPResult_p1_PD_scurve_SB, TPResult_p2_PD_scurve_SB])
print("PD Scurve MB", [TPResult_p1_PD_scurve_MB, TPResult_p2_PD_scurve_MB])
print("SH tanh SB", [TPResult_p1_SH_tanh_SB, TPResult_p2_PD_tanh_SB])
print("SH tanh MB", [TPResult_p1_SH_tanh_MB, TPResult_p2_SH_tanh_MB])
print("SH scurve SB", [TPResult_p1_PD_scurve_SB, TPResult_p2_PD_scurve_SB])
print("SH scurve MB", [TPResult_p1_PD_scurve_MB, TPResult_p2_PD_scurve_MB])

#
#
# plt.scatter(p1_rewards_PD_MB_tanh, p2_rewards_PD_MB_tanh, s=0.1, color='C0', label='Rewards')
# plt.title("PD Data Sharing Rewards: Mutual Benefits with tanh")
# plt.scatter(TPResult_p1_PD_tanh_MB, TPResult_p2_PD_tanh_MB, color='r', label='Threat point')
# print("Threat point reward found for PD tanh with MB at:", [TPResult_p1_PD_tanh_MB, TPResult_p2_PD_tanh_MB])
# plt.show()
#
# plt.scatter(p1_rewards_PD_SB_tanh, p2_rewards_PD_SB_tanh, s=0.1, color='C0', label='Rewards')
# plt.title("PD Data Sharing Rewards: Singular Benefits with tanh")
# plt.scatter(TPResult_p1_PD_tanh_SB, TPResult_p2_PD_tanh_SB, color='r', label='Threat point')
# print("Threat point reward found for PD tanh with SB at:", [TPResult_p1_PD_tanh_SB, TPResult_p2_PD_tanh_SB])
# plt.show()
#
# plt.scatter(p1_rewards_PD_MB_curves, p2_rewards_PD_MB_curves, s=0.1, label='Rewards')
# plt.title("PD Data Sharing Rewards: Mutual Benefits with scurve")
# plt.scatter(TPResult_p1_PD_scurve_MB, TPResult_p2_PD_scurve_MB, color='r', label='Threat point')
# print("Threat point reward found for PD scurve with MB at:", [TPResult_p1_PD_scurve_MB, TPResult_p2_PD_scurve_MB])
# plt.show()
#
# plt.scatter(p1_rewards_PD_SB_curves, p2_rewards_PD_SB_curves, s=0.1, label='Rewards')
# plt.title("PD Data Sharing Rewards: Singular Benefits with scurve")
# plt.scatter(TPResult_p1_PD_scurve_SB, TPResult_p2_PD_scurve_SB, color='r', label='Threat point')
# print("Threat point reward found for PD scurve with SB at:", [TPResult_p1_PD_scurve_SB, TPResult_p2_PD_scurve_SB])
# plt.show()
#
#
# plt.scatter(p1_rewards_SH_MB_tanh, p2_rewards_SH_MB_tanh, s=0.1, color='C0', label='Rewards')
# plt.title("SH Data Sharing Rewards: Mutual Benefits with tanh")
# plt.scatter(TPResult_p1_SH_tanh_MB, TPResult_p2_SH_tanh_MB, color='r', label='Threat point')
# print("Threat point reward found for SH tanh with MB at:", [TPResult_p1_SH_tanh_MB, TPResult_p2_SH_tanh_MB])
# plt.show()
#
# plt.scatter(p1_rewards_SH_SB_tanh, p2_rewards_SH_SB_tanh, s=0.1, color='C0', label='Rewards')
# plt.title("SH Data Sharing Rewards: Singular Benefits with tanh")
# plt.scatter(TPResult_p1_SH_tanh_SB, TPResult_p2_SH_tanh_SB, color='r', label='Threat point')
# print("Threat point reward found for SH tanh with SB at:", [TPResult_p1_SH_tanh_SB, TPResult_p2_SH_tanh_SB])
# plt.show()
#
# plt.scatter(p1_rewards_SH_MB_curves, p2_rewards_SH_MB_curves, s=0.1, label='Rewards')
# plt.title("SH Data Sharing Rewards: Mutual Benefits with scurve")
# plt.scatter(TPResult_p1_SH_scurve_MB, TPResult_p2_SH_scurve_MB, color='r', label='Threat point')
# print("Threat point reward found for SH scurve with MB at:", [TPResult_p1_SH_scurve_MB, TPResult_p2_SH_scurve_MB])
# plt.show()
#
# plt.scatter(p1_rewards_SH_SB_curves, p2_rewards_SH_SB_curves, s=0.1, label='Rewards')
# plt.title("SH Data Sharing Rewards: Singular Benefits with scurve")
# plt.scatter(TPResult_p1_SH_scurve_SB, TPResult_p2_SH_scurve_SB, color='r', label='Threat point')
# print("Threat point reward found for SH scurve with SB at:", [TPResult_p1_SH_scurve_SB, TPResult_p2_SH_scurve_SB])
# plt.show()


## HERE BELOW WE CAN COMPUTE THE PARETO EFFICIENT LINE

# print(rewards_p1_PD)
# print(rewards_p2_PD)
#
# print(np.sum([rewards_p1_PD, rewards_p2_PD], axis=0))