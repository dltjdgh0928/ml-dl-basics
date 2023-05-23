import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)
# y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


plt.plot(x, y)
plt.grid()
plt.show()
