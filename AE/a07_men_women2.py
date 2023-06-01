from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, UpSampling2D
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing import image
path = 'c:/_data/men_women/'
v = 'c:/_data/suit/증명사진.jpg'

target_x = 100
target_y = 100

img = image.load_img(v, target_size=(target_x, target_y))
img = image.img_to_array(img)/255.
img = np.expand_dims(img, axis=0)

datagen = ImageDataGenerator(rescale=1./255)



men_woman = datagen.flow_from_directory(path, target_size=(target_x, target_y), batch_size=5000, class_mode='binary', color_mode='rgb', shuffle=True)

men_woman_x = men_woman[0][0]
men_woman_y = men_woman[0][1]

men_woman_x_train, men_woman_x_test, men_woman_y_train, men_woman_y_test = train_test_split(men_woman_x, men_woman_y, train_size=0.7, shuffle=True, random_state=123)

x_train_noised = men_woman_x_train + np.random.normal(0, 0.3, size=men_woman_x_train.shape)
x_test_noised = men_woman_x_test + np.random.normal(0, 0.3, size=men_woman_x_test.shape)

x_train_noised = np.clip(x_train_noised, 0, 1)
x_test_noised = np.clip(x_test_noised, 0, 1)

def autoencoder():
    model = Sequential()
    # 인코더
    model.add(Conv2D(256, 3, activation='relu', padding='same', input_shape=(target_x, target_y, 3)))
    
    # 디코더
    model.add(Conv2D(256, 3, activation='relu', padding='same'))
    
    model.add(Conv2D(3, 3, activation='relu', padding='same'))
    model.summary()
    return model


model = autoencoder()

model.compile(optimizer='adam', loss='mse')
model.fit(x_train_noised, men_woman_x_train, epochs=10, batch_size=128, validation_split=0.2)

# 4. 평가, 예측
decoded_img = model.predict(x_test_noised)
img_noise = img + np.random.normal(0, 0.3, size=img.shape)
img_pred = model.predict(img)

# fig, ((ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16, ax17, ax18)) = plt.subplots(3, 6, figsize=(20, 7))

# random_images = random.sample(range(decoded_img.shape[0]), 5)
        
# for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
#     print(ax1)
#     print(type(ax1))
#     print(ax2)
#     ax.imshow(men_woman_x_test[random_images[i]].reshape(target_x, target_y, 3))
#     if i == 0:
#         ax.set_ylabel('INPUT', size=20)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])

# for i, ax in enumerate([ax6]):
#     ax.imshow(img.reshape(target_x, target_y, 3))
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
    
# for i, ax in enumerate([ax7, ax8, ax9, ax10, ax11]):
#     ax.imshow(x_test_noised[random_images[i]].reshape(target_x, target_y, 3))
#     if i == 0:
#         ax.set_ylabel('NOISE', size=20)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
    
# for i, ax in enumerate([ax12]):
#     ax.imshow(img_noise.reshape(target_x, target_y, 3))
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
    
# for i, ax in enumerate([ax13, ax14, ax15, ax16, ax17]):
#     ax.imshow(decoded_img[random_images[i]].reshape(target_x, target_y, 3))
#     if i == 0:
#         ax.set_ylabel('OUTPUT', size=20)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
    
# for i, ax in enumerate([ax18]):
#     ax.imshow(img_pred.reshape(target_x, target_y, 3))
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.tight_layout()
# plt.show()






fig, (ax6, ax12, ax18) = plt.subplots(3, 1, figsize=(20, 7))

# random_images = random.sample(range(decoded_img.shape[0]), 5)
        
# for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
#     ax.imshow(men_woman_x_test[random_images[i]].reshape(target_x, target_y, 3))
#     if i == 0:
#         ax.set_ylabel('INPUT', size=20)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])

for i, ax in enumerate([ax6]):
    print(ax6)
    print(type(ax6))
    ax.imshow(img.reshape(target_x, target_y, 3))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
# for i, ax in enumerate([ax7, ax8, ax9, ax10, ax11]):
#     ax.imshow(x_test_noised[random_images[i]].reshape(target_x, target_y, 3))
#     if i == 0:
#         ax.set_ylabel('NOISE', size=20)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
    
for i, ax in enumerate([ax12]):
    ax.imshow(img_noise.reshape(target_x, target_y, 3))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
# for i, ax in enumerate([ax13, ax14, ax15, ax16, ax17]):
#     ax.imshow(decoded_img[random_images[i]].reshape(target_x, target_y, 3))
#     if i == 0:
#         ax.set_ylabel('OUTPUT', size=20)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
    
for i, ax in enumerate([ax18]):
    ax.imshow(img_pred.reshape(target_x, target_y, 3))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()