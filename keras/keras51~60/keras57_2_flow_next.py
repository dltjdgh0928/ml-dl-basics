from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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

augment_size = 100

print(x_train.shape)        # (60000, 28, 28)
print(x_train[0].shape)     # (28, 28)
print(x_train[1].shape)     # (28, 28)
print(x_train[0][0].shape)     # (28,)
print(x_train[0][27].shape)     # (28,)

x = np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1)
print(x.shape)
x = (np.tile(x_train[0], augment_size))
print(x.shape)

print(np.zeros(augment_size))
print(np.zeros(augment_size).shape)     # (100,)

x_data = train_datagen.flow(np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),        # x 데이터
                            np.zeros(augment_size),         # y데이터 : 그림만 그리므로 불필요, 0대입                            
                            batch_size=augment_size,
                            shuffle=True
).next()


################################## .next 사용 #############################
print(x_data)       # x와 y가 합쳐진 데이터 출력
print(type(x_data))                         # <class 'tuple'>
print(x_data[0].shape)                      # (100, 28, 28, 1)
print(x_data[1].shape)                      # (100,)
print(x_data[0][0].shape)                   # (28, 28, 1)
print(x_data[0][99].shape)                  # (28, 28, 1)
print(x_data[0][0][0].shape)                # (28, 1)
print(x_data[0][0][0][0].shape)             # (1,)
print(x_data[0][0][0][0][0].shape)          # ()
print(type(x_data[0]))

import matplotlib.pyplot as plt
plt.figure(figsize=(7, 7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')
plt.show()



################################## .next 미사용 #############################
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









 