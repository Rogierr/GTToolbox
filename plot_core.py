import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

x = np.arange(0, 9)
y = np.arange(0, 9)
z = np.arange(0, 9)

Axes3D.plot(x, y, z)