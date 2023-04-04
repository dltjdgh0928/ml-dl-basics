import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# 1. 데이터
docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글세요', 
        '별로에요', '생각보다 지루해요', '연기가 어색해요', 
        '재미없어요', '너무 재미없다', ' 참 재밋네요', '환희가 잘 생기긴 했어요',
        '환희가 안해요']

# 긍정인지 부정인지 맞춰봐!!!

x_predict = ['나는 성호가 정말 재미없다 너무 정말']

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])
# 긍정 1, 부정 0

data = docs + x_predict

token = Tokenizer()
token.fit_on_texts(data)
data = token.texts_to_sequences(data)

print(data)
# [[1, 7], [2, 8], [2, 3, 9, 10], [11, 12, 13], [14, 15, 16, 17, 18], [19], [20], [21, 22], [23, 24], [25], [1, 4], [2, 26], [5, 3, 27, 28], [5, 29], [30, 31, 6, 4, 1, 6]]
print(max(len(i) for i in data))
print(sum(map(len, data))/len(data))
print(sum(len(i) for i in data)/len(data))

data_pad = pad_sequences(data, maxlen=5, padding='pre', truncating='pre')
print(data_pad)

pad_x_train = data_pad[:14, :]
pad_x_pred = data_pad[14:, :]
print(pad_x_train.shape)
print(pad_x_pred.shape)
word_index = len(token.word_index)

# 2. 모델
model = Sequential()
model.add(Embedding(word_index, 32, input_length=5))
model.add(LSTM(32))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(pad_x_train, labels, epochs=100, batch_size=64, validation_split=0.2)

# 4. 평가, 예측
acc = model.evaluate(pad_x_train, labels)[1]
print('acc :', acc)

y_pred = model.predict(pad_x_pred)
print('predict : ', y_pred)