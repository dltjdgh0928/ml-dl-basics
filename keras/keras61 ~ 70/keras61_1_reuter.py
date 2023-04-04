from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer

# 1. 데이터
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)

print(x_train)
print(y_train)    # [ 3 4 3 ... 25 3 25 ]
print(x_train.shape, y_train.shape)          # (8982,) (8982,)
print(x_test.shape, y_test.shape)            # (2246,) (2246,)

print(len(x_train[0]), len(x_train[1]))      # 87 56
print(np.unique(y_train))                    # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 ]

# softmax 46, embedding input_dim=10000, output_dim=마음대로, input_length=max(len)


data = docs + x_predict
print(data)

token = Tokenizer()
token.fit_on_texts(data)
print(token.word_index)
print(token.word_counts)

x = token.texts_to_sequences(data)
print(x)
# [[2, 5], [1, 6], [1, 3, 7, 8], [9, 10, 11], [12, 13, 14, 15, 16], [17], [18], [19, 20], [21, 22], [23], [2, 24], [1, 25], [4, 3, 26, 27], [4, 28]]

# [[1, 7], [2, 8], [2, 3, 9, 10], [11, 12, 13], [14, 15, 16, 17, 18], [19], [20], [21, 22], [23, 24], [25], [1, 4], [2, 26], [5, 3, 27, 28], [5, 29], [30, 31, 6, 4, 1, 6]]

pad_x = pad_sequences(x, padding='pre', maxlen=5)
print(pad_x.shape)      # (15, 5)
pad_x_train = pad_x[:14, :]
pad_x_pred = pad_x[14, :]
pad_x_train = pad_x_train.reshape(pad_x_train.shape[0], pad_x_train.shape[1], 1)
pad_x_pred = pad_x_pred.reshape(1, 5, 1)

word_index = len(token.word_index)
print("단어사전의 갯수 : ", word_index)

# 2. 모델
model = Sequential()
model.add(Embedding(word_index, 32,'uniform',None,None,None,False,5))
# model.add(Embedding(28, 32, 5))           # error 
# model.add(Embedding(input_dim=28, output_dim=33, input_length=5))
# model.add(Reshape(target_shape=(5, 1), input_shape=(5,)))
# model.add(Dense(32, input_shape=(5,)))
model.add(LSTM(32))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(pad_x_train, labels, epochs=100, batch_size=16)

# 4. 평가, 예측
acc = model.evaluate(pad_x_train, labels)[1]
print('acc : ', acc)

y_pred = np.round(model.predict(pad_x_pred))
print(y_pred)