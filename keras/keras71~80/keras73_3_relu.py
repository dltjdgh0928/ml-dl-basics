import numpy as np
import matplotlib.pyplot as plt
np.random.seed(337)

def relu(x):
    return np.maximum(0, x)

def elu(x):
    return np.where(x > 0, x, 1 * (np.exp(x) - 1))

def selu(x):
    return np.where(x > 0, x, np.random.uniform(0, 1) * (np.exp(x) - 1))

def LeakyReLu(x):
    return np.maximum(0.01*x, x)

x = np.arange(-5, 5, 0.1)

y_relu = relu(x)
y_elu = elu(x)
y_selu = selu(x)
y_LeakyReLu = LeakyReLu(x)

plt.subplot(2, 2, 1)
plt.plot(x, y_relu)
plt.title('RELU')
plt.grid()
plt.subplot(2, 2, 2)
plt.plot(x, y_elu)
plt.title('ELU')
plt.grid()
plt.subplot(2, 2, 3)
plt.plot(x, y_selu)
plt.title('SELU')
plt.grid()
plt.subplot(2, 2, 4)
plt.plot(x, y_LeakyReLu)
plt.title('LeakyReLu')
plt.grid()
plt.show()
