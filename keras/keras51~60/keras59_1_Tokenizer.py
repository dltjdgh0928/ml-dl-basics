from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index) 

# 정렬기준 : 최빈값 -> 앞에서부터

print(token.word_counts)

x = token.texts_to_sequences([text])    
print(x)        # [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]]

######### 1. to_categorical ##########
# from tensorflow.keras.utils import to_categorical
# x = to_categorical(x)
# print(x)
# print(x.shape)          # (1, 11, 9)


######### 2. get_dummies ##########
# import pandas as pd
# import numpy as np

# # x = pd.get_dummies(np.array(x).reshape(11,))
# x = pd.get_dummies(np.array(x).ravel())
# print(x)

# x = pd.get_dummies(x[0])
# print(x)



######### 3. 사이킷런 onehot ##########
from sklearn.preprocessing import OneHotEncoder
import numpy as np
ohe = OneHotEncoder()
x = ohe.fit_transform(np.array(x).reshape(-1, 1)).toarray()
print(x)