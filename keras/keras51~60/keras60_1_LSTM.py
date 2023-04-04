from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape
import numpy as np

# 1. 데이터
docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글세요', 
        '별로에요', '생각보다 지루해요', '연기가 어색해요', 
        '재미없어요', '너무 재미없다', ' 참 재밋네요', '환희가 잘 생기긴 했어요',
        '환희가 안해요']


# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '잘': 3, '환희가': 4, '재밋어요': 5, '최고에요': 6, 
# '만든': 7, '영화예요': 8, '추천하고': 9, '싶은': 10, '영화입니다': 11, ' 
# 한': 12, '번': 13, '더': 14, '보고': 15, '싶네요': 16, '글세요': 17, '별 
# 로에요': 18, '생각보다': 19, '지루해요': 20, '연기가': 21, '어색해요': 22, '재미없어요': 23, '재미없다': 24, '재밋네요': 25, '생기긴': 26, '했어요
# ': 27, '안해요': 28}


print(token.word_counts)
# OrderedDict([('너무', 2), ('재밋어요', 1), ('참', 3), ('최고에요', 1), ('잘', 2), ('만든', 1), ('영화예요', 1), ('추천하고', 1), ('싶은', 1), ('영
# 화입니다', 1), ('한', 1), ('번', 1), ('더', 1), ('보고', 1), ('싶네요', 1), ('글세요', 1), ('별로에요', 1), ('생각보다', 1), ('지루해요', 1), ('연
# 기가', 1), ('어색해요', 1), ('재미없어요', 1), ('재미없다', 1), ('재밋네 
# 요', 1), ('환희가', 2), ('생기긴', 1), ('했어요', 1), ('안해요', 1)])

x = token.texts_to_sequences(docs)
print(x)
# [[2, 5], [1, 6], [1, 3, 7, 8], [9, 10, 11], [12, 13, 14, 15, 16], [17],
#  [18], [19, 20], [21, 22], [23], [2, 24], [1, 25], [4, 3, 26, 27], [4, 28]]

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5)
# pre, post
print(pad_x)
print(pad_x.shape)      # (14, 5)
pad_x = pad_x.reshape(14, 5, 1)
word_index = len(token.word_index)
print("단어사전의 갯수 : ", word_index)

# 2. 모델
model = Sequential()
model.add(LSTM(32, input_shape=(5, 1)))
# model.add(Reshape(target_shape=(5, 1), input_shape=(5,)))
# model.add(Dense(32, input_shape=(5,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(pad_x, labels, epochs=100, batch_size=16)

# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print('acc : ', acc)
print(pad_x[0])
x_pred= pad_x[0].reshape(1, 5, 1)
y_pred_1 = np.round(model.predict(x_pred))
print(y_pred_1)