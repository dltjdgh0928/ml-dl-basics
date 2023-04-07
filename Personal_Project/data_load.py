import os
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
from tensorflow.keras.callbacks import EarlyStopping


# 1. 데이터
def bring(path):
    data_list = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            with open(os.path.join(path, filename), 'r') as f:
                data = f.read().replace('\n', '').replace(' ', '')
            data = re.sub(r"[0-9]", "", data)
            data = data.replace('a', '1').replace('t', '2').replace('c', '3').replace('g', '4').replace('y', '5').replace('w', '6').replace('r', '7').replace('k', '8').replace('v', '9').replace('n', '10').replace('s', '11')
            data = np.array([int(i) for i in data])
            data = data.reshape(1, -1)
            data = pad_sequences(data, maxlen=700, padding='pre', truncating='pre')
            data = data.reshape(-1, 1)
            data_list.append(data)
    all_data = np.array(np.concatenate(data_list, axis=1))
    all_data = all_data.T
    return all_data

homo_path = './_data/pp/homo_sapiens/'
x_homo = bring(homo_path)
y_homo = np.array([0]*x_homo.shape[0])
print(y_homo)
print(y_homo.shape)
culex_path = './_data/pp/Culex/'
x_culex = bring(culex_path)
y_culex = np.array([1]*x_culex.shape[0])

x = np.concatenate([x_homo, x_culex], axis=0)
y = np.concatenate([y_homo, y_culex], axis=0)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)


# 2. 모델
model = Sequential()
model.add(Dense(32, input_shape=(700,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='acc', patience=100, mode='auto', restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=3, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print('acc : ', acc)

random_index = np.random.randint(0, x_test.shape[0])
x_pred = x[random_index].reshape(1, -1)
y_pred = np.round(model.predict(x_pred))
print('Real Speices : ', y[random_index], 'Predict Speices : ', y_pred)
