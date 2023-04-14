import numpy as np
from sklearn.datasets import load_breast_cancer

# 1. 데이터
# datasets = load_iris()
# x = datasets.data
# y = datasets['target']
x, y = load_breast_cancer(return_X_y=True)

print(x.shape, y.shape)     # (150, 4) (150,)

# 2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,  RandomTreesEmbedding



model = LinearSVC()
# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))

model = LogisticRegression()
# model = DecisionTreeRegressor()
# model = DecisionTreeClassifier()
# model = RandomForestRegressor()

# 3. 컴파일, 훈련
# model.compile(loss ='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x, y, epochs=100, validation_split=0.2)
model.fit(x, y)

# 4. 평가, 예측
# results = model.evaluate(x, y)
results = model.score(x, y)



print(results)


