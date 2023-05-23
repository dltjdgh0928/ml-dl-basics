from tensorflow.keras.datasets import mnist, cifar100
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import time

tf.random.set_seed(337)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', input_shape=(x_train.shape[1], x_train.shape[2],3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2), padding='valid', activation='relu'))
model.add(Conv2D(32, 2))
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_acc', mode='auto', patience=5, verbose=1, restore_best_weights=True)

start = time.time()
hist = model.fit(x_train, y_train, epochs=150, batch_size=128, validation_split=0.368, callbacks=[es])
end = time.time()

# 4. 평가, 예측
print(f'result : {model.evaluate(x_test, y_test)}\nloss : {model.evaluate(x_test, y_test)[0]}\nacc : {accuracy_score(np.argmax(y_test,axis=1), np.argmax(model.predict(x_test),axis=1))}\ntime used : {int((end - start)//60)}m {round((end - start)%60, 3)}s')

# result :  [2.7789909839630127, 0.3100999891757965]
# loss : 2.7789909839630127
# acc 0.3100999891757965
# acc : 0.3101
# time used : 0m 36.228s