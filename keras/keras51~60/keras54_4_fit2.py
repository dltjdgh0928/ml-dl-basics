import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                #    horizontal_flip=True, vertical_flip=True, width_shift_range=0.1, height_shift_range=0.1,
                                #    rotation_range=5, zoom_range=1.2, shear_range=0.7, fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory('d:/study_data/_data/brain/train/', target_size=(100, 100), batch_size=5, class_mode='binary', 
                                  color_mode='grayscale', shuffle=True)

# Found 160 images belonging to 2 classes. ( 전체 데이터 쓸려면 batch_size 160 이상 )

xy_test = test_datagen.flow_from_directory('d:/study_data/_data/brain/test/', target_size=(100, 100), batch_size=5, class_mode='binary', 
                                  color_mode='grayscale', shuffle=True)

# Found 120 images belonging to 2 classes.

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000002943E405E20>

print(xy_train[0])
print(len(xy_train))            # 32
print(len(xy_train[0]))         # 2

print(xy_train[0][0])           # 엑스 5개
print(xy_train[0][1])           # [1. 0. 0. 0. 1.]
print(xy_train[0][0].shape)     # (5, 200, 200, 1) : grayscale
# print(xy_train[0][0].shape)     # (5, 200, 200, 3) : rgb
# print(xy_train[0][0].shape)     # (5, 200, 200, 4) : rgba
print(xy_train[0][1].shape)     # (5,)

print(type(xy_train))           # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))        # <class 'tuple'>
print(type(xy_train[0][0]))     # <class 'numpy.ndarray'>
print(type(xy_train[0][1]))     # <class 'numpy.ndarray'>

# 현재 x는 (5, 200, 200, 1) 짜리 데이터가 32덩이

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(100, 100, 1), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(xy_train[:][0], xy_train[:][1], epochs=10)        # TypeError: '>=' not supported between instances of 'slice' and 'int'
# hist = model.fit(xy_train[0][0], xy_train[0][1], epochs=10, batch_size=16, validation_data=(xy_test[0][0], xy_test[0][1]))        # 통배치면 요거 가능

# hist = model.fit_generator(xy_train, epochs=30, steps_per_epoch=32, validation_data=xy_test, validation_steps=24)
hist = model.fit(xy_train, epochs=30, steps_per_epoch=32, validation_data=xy_test, validation_steps=24)

# steps_per_epoch = 전체데이터/batch = 160/5 = 32
# validation_steps = 발리데이터/batch = 120/5 = 24

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('loss :', loss[-1])
print('val_loss : ', val_loss[-1])
print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.subplot(1, 2, 1)
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(acc, label='acc')
plt.plot(val_acc, label='val_acc')
plt.grid()
plt.legend()
plt.show()