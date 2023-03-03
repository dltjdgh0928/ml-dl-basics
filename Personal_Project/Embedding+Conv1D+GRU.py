import os
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Bidirectional, Dropout, SimpleRNN, Embedding, Conv1D, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# 1. 데이터
timesteps=3
def split_x(data, timesteps):
    a = []
    for i in range(len(data)-timesteps+1):
        b = data[i:(i+timesteps)]
        a.append(b)
    return np.array(a)
        
def bring(path):
    data_list = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            with open(os.path.join(path, filename), 'r') as f:
                data = f.read().replace('\n', '').replace(' ', '')
            data = re.sub(r"[0-9]", "", data)
            data = data.replace('a', '1').replace('t', '2').replace('c', '3').replace('g', '4').replace('y', '5').replace('w', '6')\
                .replace('r', '7').replace('k', '8').replace('v', '9').replace('n', '10').replace('s', '11').replace('m', '12')
            data = np.array([int(i) for i in data])
            data = data.reshape(1, -1)
            data = pad_sequences(data, maxlen=maxlen, padding='pre', truncating='pre')
            data = data.reshape(-1, 1)
            data = split_x(data, timesteps)
            data = data.reshape(1, -1, timesteps)
            data_list.append(data)
    all_data = np.array(np.concatenate(data_list, axis=0))
    return all_data

maxlen=800

homo_path = './_data/pp/homo_sapiens/'
x_homo = bring(homo_path)
y_homo = np.array([0]*x_homo.shape[0])

culex_path = './_data/pp/Culex/'
x_culex = bring(culex_path)
y_culex = np.array([1]*x_culex.shape[0])

haemagogus_path = './_data/pp/Haemagogus/'
x_haemagogus = bring(haemagogus_path)
y_haemagogus = np.array([2]*x_haemagogus.shape[0])

ovis_path = './_data/pp/Ovis_aries/'
x_ovis = bring(ovis_path)
y_ovis = np.array([3]*x_ovis.shape[0])

mus_path = './_data/pp/Mus_musculus/'
x_mus = bring(mus_path)
y_mus = np.array([4]*x_mus.shape[0])

sciuridae_path = './_data/pp/Sciuridae/'
x_sciuridae = bring(sciuridae_path)
y_sciuridae = np.array([5]*x_sciuridae.shape[0])

canis_path = './_data/pp/Canis_lupus/'
x_canis = bring(canis_path)
y_canis = np.array([6]*x_canis.shape[0])

vulpes_path = './_data/pp/Vulpes_vulpes/'
x_vulpes = bring(vulpes_path)
y_vulpes = np.array([7]*x_vulpes.shape[0])

sus_path = './_data/pp/Sus_scrofa/'
x_sus = bring(sus_path)
y_sus = np.array([8]*x_sus.shape[0])

x = np.concatenate([x_homo, x_culex, x_haemagogus, x_ovis, x_mus, x_sciuridae, x_canis, x_vulpes, x_sus], axis=0)
y = np.concatenate([y_homo, y_culex, y_haemagogus, y_ovis, y_mus, y_sciuridae, y_canis, y_vulpes, y_sus], axis=0)

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)


# 2. 모델
model = Sequential()
model.add(Embedding(input_dim=9, output_dim=timesteps, input_shape=((maxlen-timesteps+1),)))
model.add(Conv1D(64, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256))
model.add(Dropout(0.1))
model.add(Dense(64))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(y[0]), activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_acc', patience=100, mode='auto', restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=3, validation_split=0.2, callbacks=[es])


# 4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print('[ 개체수 ] \nhomo sapiens :', x_homo.shape[0], '\nCulex : ', x_culex.shape[0], '\nHaemagogus : ', x_haemagogus.shape[0],\
                 '\nOvis aries :', x_ovis.shape[0], '\nMus musculus : ', x_mus.shape[0],\
                 '\nSciuridae : ', x_sciuridae.shape[0], '\nCanis lupus : ', x_canis.shape[0],\
                 '\nVulpes vulpes : ', x_vulpes.shape[0], '\nSus crofa : ', x_sus.shape[0])

print('\nacc : ', acc)

random_index = np.random.randint(0, x.shape[0])
x_pred = x[random_index].reshape(1, -1, timesteps)
y_pred = np.argmax(model.predict(x_pred), axis=1)

def Speices(x):
    index=['Homo Sapiens','Culex','Haemagogus', 'Ovis aries', 'Mus musculus', 'Sciuridae', 'Canis lupus', 'Vulpes vulpes', 'Sus crofa']
    for i in range(len(index)):
        if x == i:
            return index[i]
    return False


print('\nRandom Real Speices : ', Speices(np.argmax(y[random_index].reshape(1, -1), axis=1)), '\nPredict Speices : ', Speices(y_pred))
