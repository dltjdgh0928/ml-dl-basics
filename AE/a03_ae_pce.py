import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 1, size=x_test.shape)

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

model = autoencoder(1)
model.compile(optimizer='adam', loss='mse')
model.fit(x_train_noised, x_train, epochs=30, batch_size=128, validation_split=0.2)

# 4. 평가, 예측
decoded_img = np.round(model.predict(x_test_noised))
# decoded_img = autoencoder.predict(x_test_noised)
import matplotlib.pyplot as plt
n=10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noised[i].reshape(28, 28))
    plt.gray()
    # ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
