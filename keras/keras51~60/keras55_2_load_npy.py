import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. 데이터
path = 'd:/study_data/_save/_npy/'
# np.save(path + 'keras55_1_x_train.npy', arr=xy_train[0][0])
# np.save(path + 'keras55_1_x_test.npy', arr=xy_test[0][0])
# np.save(path + 'keras55_1_y_train.npy', arr=xy_train[0][1])
# np.save(path + 'keras55_1_y_test.npy', arr=xy_test[0][1])

x_train = np.load(path + 'keras55_1_x_train.npy')
x_test = np.load(path + 'keras55_1_x_test.npy')
y_train = np.load(path + 'keras55_1_y_train.npy')
y_test = np.load(path + 'keras55_1_y_test.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

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
hist = model.fit(x_train, y_train, epochs=30, steps_per_epoch=32, validation_split=0.2, validation_steps=24)

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

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
print('r2 : ', r2_score(y_test, y_predict))