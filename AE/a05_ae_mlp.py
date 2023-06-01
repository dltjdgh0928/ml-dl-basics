import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
seed=337
tf.random.set_seed(seed)
np.random.seed(seed)

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

print(np.random.normal(0, 1, size=x_train.shape))

print(x_train_noised.shape, x_test_noised.shape)
print(np.max(x_train_noised), np.min(x_train_noised))
print(np.max(x_test_noised), np.min(x_test_noised))

x_train_noised = np.clip(x_train_noised, 0, 1)
x_test_noised = np.clip(x_test_noised, 0, 1)
print(np.max(x_train_noised), np.min(x_train_noised))       # 1.0   0.0
print(np.max(x_test_noised), np.min(x_test_noised))         # 1.0   0.0

def autoencoder(a, b, c, d, e):
    model = Sequential()
    model.add(Dense(units=a, input_shape=(784,)))
    model.add(Dense(units=b))
    model.add(Dense(units=c))
    model.add(Dense(units=d))
    model.add(Dense(units=e))
    model.add(Dense(784, activation='sigmoid'))
    return model


model = autoencoder(784, 784, 784, 784, 784)        # 100
# model = autoencoder(486)        # 99.9
# model = autoencoder(331)        # 99%
# model = autoencoder(154)        # pca 95% 

# print(np.argmax(cumsum >= 0.95)+1)        # 154
# print(np.argmax(cumsum >= 0.99)+1)        # 331
# print(np.argmax(cumsum >= 0.999)+1)        # 486
# print(np.argmax(cumsum >= 1.0)+1)        # 713

model.compile(optimizer='adam', loss='mse')
model.fit(x_train_noised, x_train, epochs=30, batch_size=128, validation_split=0.2)

# 4. 평가, 예측
decoded_img = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20, 7))

random_images = random.sample(range(decoded_img.shape[0]), 5)
        
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('NOISE', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_img[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()
