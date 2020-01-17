"""The two singular benefits functions for the Data Sharing Dilemma"""

def sb_function_p1(x):

    sb = float(2/3)*x[:,0] + x[:,1] + float(1/3)*x[:,2]

    return sb


def sb_function_p2(x):

    sb = float(2 / 3) * x[:, 0] + float(1 / 3) * x[:, 1] + x[:, 2]

    return sb