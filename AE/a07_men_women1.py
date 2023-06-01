from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, UpSampling2D
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random

path = 'c:/_data/men_women/'

datagen = ImageDataGenerator(rescale=1./255)

target_x = 100
target_y = 100

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
    model.add(Conv2D(128, 3, activation='relu', padding='same', input_shape=(target_x, target_y, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D())
    
    # 디코더
    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, 2, activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    
    model.add(Conv2D(3, 3, activation='relu', padding='same'))
    model.summary()
    return model


model = autoencoder()

model.compile(optimizer='adam', loss='mse')
model.fit(x_train_noised, men_woman_x_train, epochs=30, batch_size=128, validation_split=0.2)

# 4. 평가, 예측
decoded_img = model.predict(x_test_noised)



fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20, 7))

random_images = random.sample(range(decoded_img.shape[0]), 5)
        
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(men_woman_x_test[random_images[i]].reshape(target_x, target_y, 3))
    if i == 0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(target_x, target_y, 3))
    if i == 0:
        ax.set_ylabel('NOISE', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_img[random_images[i]].reshape(target_x, target_y, 3))
    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()