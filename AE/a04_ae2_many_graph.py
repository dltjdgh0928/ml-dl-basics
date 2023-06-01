import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from matplotlib import pyplot as plt
import random

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

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,)))
    model.add(Dense(784, activation='sigmoid'))
    return model

model_01 = autoencoder(1)
model_08 = autoencoder(8)
model_32 = autoencoder(32)
model_64 = autoencoder(64)
model_154 = autoencoder(154)
model_331 = autoencoder(331)
model_486 = autoencoder(486)
model_713 = autoencoder(713)

model_list = [model_01, model_08, model_32, model_64, model_154, model_331, model_486, model_713]

for j in range(len(model_list)):
    print(f'{j+1}')
    model_list[j].compile(optimizer='adam', loss='mse')
    model_list[j].fit(x_train_noised, x_train, epochs=5, batch_size=128, validation_split=0.2)

    globals()[f'decoded_img_{j+1}'] = np.round(model_list[j].predict(x_test_noised))
    random_images = random.sample(range(globals()[f'decoded_img_{j+1}'].shape[0]), 5)
    
    for k in range(5):
        plt.subplot(8, 5, 5*j+1+k)
        plt.plot(globals()[f'decoded_img_{j+1}'][random_images[k]])
plt.show()
