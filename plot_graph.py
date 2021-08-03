import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 100000, endpoint=False)
y = np.multiply(1.1, (1 - np.divide(0.1, x + 0.1)))

plt.plot(x,y)
plt.title("Function mu(rho) with a=0.1 and b=1.1")
plt.ylabel("Prediction Accuracy")
plt.xlabel("Weighted Learning Rate")
plt.savefig("figures/plot_learning_curve.png", dpi=300, bbox_inches="tight")
plt.show()