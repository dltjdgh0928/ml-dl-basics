from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()
# token.fit_on_texts(text)
# print(token.word_index)
# # {'마': 1, '구': 2, '는': 3, '매': 4, '우': 5, '나': 6, '진': 7, '짜': 8, '맛': 9, '있': 10, '밥': 11, '을': 12, '엄': 13, '청': 14, '먹': 15, '었': 16, '다': 17}
# print(token.word_counts)
# OrderedDict([('나', 1), ('는', 2), ('진', 1), ('짜', 1), ('매', 2), ('우', 2), ('맛', 1), ('있', 1), ('밥', 1), ('을', 1), ('엄', 1), ('청', 1), ('마', 3), ('구', 3), ('먹', 1), ('었', 1), ('다', 1)])
token.fit_on_texts([text])
print(token.word_index)
# {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
print(token.word_counts)
# OrderedDict([('나는', 1), ('진짜', 1), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])

x = token.texts_to_sequences([text])

# 1. to_categorical
# from tensorflow.keras.utils import to_categorical
# x = to_categorical(x)
# print(x)
# print(x.shape)

# x = x[:, :, 1:]
# print(x)
# print(x.shape)



# 2. get_dummies
# import pandas as pd
# import numpy as np
# # x = pd.get_dummies(np.array(x).reshape(-1,))
# x = pd.get_dummies(np.array(x).ravel())

# print(x)



# 3. 사이킷런
# from sklearn.preprocessing import OneHotEncoder
# import numpy as np
# ohe = OneHotEncoder()
# x = ohe.fit_transform(np.array(x).reshape(-1, 1)).toarray()
# print(x)