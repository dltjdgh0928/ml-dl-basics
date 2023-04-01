from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras import regularizers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 1. 데이터
path_save = 'd:/study_data/_save/breed/'

breed_x = np.load(path_save + 'breed_x.npy')
breed_y = np.load(path_save + 'breed_y.npy')

breed_x_train, breed_x_test, breed_y_train, breed_y_test = train_test_split(breed_x, breed_y, train_size=0.7, shuffle=True, random_state=123)

# 2. 모델
# model = Sequential()
# model.add(Conv2D(64, 2, input_shape=(150, 150, 4), activation='relu'))
# model.add(Conv2D(64, 2, activation='selu'))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(5, activation='softmax'))

model = Sequential()
model.add(Conv2D(64, 2, input_shape=(1000, 1000, 4), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(64, 2, activation='selu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Flatten())
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
hist = model.fit(breed_x_train, breed_y_train, epochs=30, batch_size=64, validation_data=(breed_x_test, breed_y_test))

# 4. 평가, 예측
loss = model.evaluate(breed_x_test, breed_y_test)
print('loss :', loss)

y_pred = model.predict(breed_x_test)
print('acc : ', accuracy_score(np.argmax(breed_y_test, axis=1), np.argmax(y_pred, axis=1)))
