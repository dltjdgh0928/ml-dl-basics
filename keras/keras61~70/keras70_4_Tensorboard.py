from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import time

tf.random.set_seed(337)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(2, (2,2), padding='same', input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(2, 2))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', mode='auto', patience=20, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=10, verbose=1, factor=0.8)
tb = TensorBoard(log_dir='c:/STUDY/_save/_tensorboard/_graph',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=True)

start = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2, callbacks=[es, reduce_lr, tb])
end = time.time()


# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)
print('loss :', result[0])
print('acc', result[1])
y_predict = model.predict(x_test)
acc = accuracy_score(np.argmax(y_test,axis=1), np.argmax(y_predict,axis=1))
print(f'acc : {acc}')

# http://localhost:6006/