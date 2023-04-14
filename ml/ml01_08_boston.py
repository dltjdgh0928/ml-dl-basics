# 회귀 데이터 싹 다 모아서 테스트
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.datasets import fetch_california_housing

# 1. 데이터
# datasets = load_iris()
# x = datasets.data
# y = datasets['target']
x, y = fetch_california_housing(return_X_y=True)


# 2. 모델구성


# model = DecisionTreeRegressor()
# model = RandomForestRegressor()

# 3. 컴파일, 훈련
# model.compile(loss ='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x, y, epochs=100, validation_split=0.2)
model.fit(x, y)

# 4. 평가, 예측
# results = model.evaluate(x, y)
results = model.score(x, y)



print(results)


