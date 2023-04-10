import os
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

exon_pattern = re.compile(r"tac(?:.{3})*?(?:atc|act|att)")

# 1. 데이터
def bring(path):
    exon_list = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            with open(os.path.join(path, filename), 'r') as f:
                data = f.read().replace('\n', '').replace(' ', '')
            data = re.sub(r"[0-9]", "", data)

            exons = []
            last_stop = 0
            for match in exon_pattern.finditer(data):
                start = match.start()
                end = match.end()
                if start % 3 == 0 and start >= last_stop:
                    exons.append(match.group())
                    last_stop = end

            if exons:
                print("Exons found:")
                for exon in exons:
                    print(exon)
                print("Number of exons:", len(exons))
                
                exons = "".join(exons)
                exons = exons.replace('a', '1').replace('t', '2').replace('c', '3').replace('g', '4').replace('y', '5').replace('w', '6').replace('r', '7').replace('k', '8').replace('v', '9').replace('n', '10').replace('s', '11').replace('m', '12')
                exons = [list(str(i)) for i in exons]
                exons = np.array(exons).reshape(-1,)
                exons = pad_sequences([exons], maxlen=maxlen, padding='pre', truncating='pre')
                exons = exons.reshape(-1, 1)
                exon_list.append(exons)
            else:
                print("No exons found")
                
    all_data = np.array(np.concatenate(exon_list, axis=1))
    all_data = all_data.T
    return all_data

maxlen = 200

homo_path = './_data/pp/homo_sapiens/'
x_homo = bring(homo_path)
y_homo = np.array([0]*x_homo.shape[0])

print(x_homo)
print(x_homo.shape)

culex_path = './_data/pp/Culex/'
x_culex = bring(culex_path)
y_culex = np.array([1]*x_culex.shape[0])

haemagogus_path = './_data/pp/Haemagogus/'
x_haemagogus = bring(haemagogus_path)
y_haemagogus = np.array([2]*x_haemagogus.shape[0])

x = np.concatenate([x_homo, x_culex, x_haemagogus], axis=0)
y = np.concatenate([y_homo, y_culex, y_haemagogus], axis=0)

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)


# 2. 모델
model = Sequential()
model.add(Dense(32, input_shape=(maxlen,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(y[0]), activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='acc', patience=20, mode='auto', restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=3, validation_split=0.2, callbacks=[es])


# 4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print('[ 개체수 ] \nhomo sapiens :', x_homo.shape[0], '\nCulex : ', x_culex.shape[0], '\nHaemagogus : ', x_haemagogus.shape[0])
print('acc : ', acc)

random_index = np.random.randint(0, x.shape[0])
x_pred = x[random_index].reshape(1, -1)
y_pred = np.argmax(model.predict(x_pred), axis=1)

def Speices(x):
    index=['Homo Sapiens','Culex','Haemagogus']
    for i in range(len(index)):
        if x == i:
            return index[i]
    return False


print('Random Real Speices : ', Speices(np.argmax(y[random_index].reshape(1, -1), axis=1)), '\nPredict Speices : ', Speices(y_pred))





