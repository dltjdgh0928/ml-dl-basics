from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(y_train)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augment_size = 40000

np.random.seed(0)
# randidx = np.random.randint(60000, size=40000)
randidx = np.random.randint(x_train.shape[0], size=augment_size)

print(randidx)                              # [31741 39588  3826 ... 46791 56223  3944]
print(randidx.shape)                        # (40000,)
print(np.min(randidx), np.max(randidx))     # 0 59997

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented)
print(x_augmented.shape, y_augmented.shape)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)

# x_augmented = train_datagen.flow(x_augmented, y_augmented, batch_size=augment_size, shuffle=False)

# print(x_augmented[0][0].shape)

x_augmented = train_datagen.flow(x_augmented, y_augmented, batch_size=augment_size, shuffle=False).next()[0]
print(x_augmented.shape)
print(x_augmented[0][0].shape)
print(x_train.shape)
x_train = np.concatenate([x_train/255., x_augmented], axis=0)
print(x_train.shape)
y_train = np.concatenate([y_train, y_augmented], axis=0)

print(y_augmented.shape)
print(np.max(x_train), np.min(x_train))
print(np.max(x_augmented), np.min(x_augmented))
print(np.min(y_augmented), np.max(y_augmented))

# print(x_train.shape)        # (60000, 28, 28)
# print(x_train[0].shape)     # (28, 28)
# print(x_train[1].shape)     # (28, 28)
# print(x_train[0][0].shape)     # (28,)
# print(x_train[0][27].shape)     # (28,)

# x = np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1)
# print(x.shape)
# x = (np.tile(x_train[0], augment_size))
# print(x.shape)

# print(np.zeros(augment_size))
# print(np.zeros(augment_size).shape)     # (100,)

# x_data = train_datagen.flow(np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),        # x 데이터
#                             np.zeros(augment_size),         # y데이터 : 그림만 그리므로 불필요, 0대입                            
#                             batch_size=augment_size,
#                             shuffle=True
# )

# print(x_data)
# print(x_data[0])        # x와 y가 모두 포함
# print(x_data[0][0].shape)                          # (100, 28, 28, 1)
# print(x_data[0][1].shape)                          # (100,)
# print(x_data[0][0][0].shape)                       # (28, 28, 1)
# print(x_data[0][0][0][0].shape)                    # (28, 1)
# print(x_data[0][0][0][0][0].shape)                 # (1,)
# print(x_data[0][0][0][0][0][0].shape)              # ()

# import matplotlib.pyplot as plt
# plt.figure(figsize=(7, 7))
# for i in range(49):
#     plt.subplot(7, 7, i+1)
#     plt.axis('off')
#     plt.imshow(x_data[0][0][i], cmap='gray')
# plt.show()

# # y_data = train_datagen.flow()






