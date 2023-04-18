# 분류데이터들 싹 모아서 테스트
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, fetch_covtype, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import warnings
warnings.filterwarnings(action='ignore')


index1 = [load_iris, load_breast_cancer, load_digits, fetch_covtype, load_wine]

index2 = [LinearSVC(max_iter=100000), LogisticRegression(max_iter=100000), DecisionTreeClassifier(max_depth=1000), RandomForestClassifier(max_depth=1000)]

scaler = MinMaxScaler()

for i in index1:
    x, y = i(return_X_y=True)
    x = scaler.fit_transform(x)
    for j in index2:
        model = j
        model.fit(x, y)
        results = model.score(x, y)
        print(i.__name__, type(j).__name__, results)




