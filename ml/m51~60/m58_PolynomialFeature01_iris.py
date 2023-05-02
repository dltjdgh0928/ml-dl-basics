import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits, fetch_california_housing, load_diabetes
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, RandomForestRegressor

# 1. 데이터
data_list = [load_breast_cancer, load_iris, load_wine, load_digits, fetch_california_housing, load_diabetes]
for i in range(len(data_list)):
    x, y = data_list[i](return_X_y=True)

    pf = PolynomialFeatures(degree=2)
    x_pf = pf.fit_transform(x)
    print(x_pf)

    x_train, x_test, y_train, y_test = train_test_split(x_pf, y, random_state=123, train_size=0.8, shuffle=True, stratify=y)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 2. 모델
    if i<3:
        model = RandomForestClassifier()
    else:
        model = RandomForestRegressor()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    # 4. 평가, 예측
    print('score : ', model.score(x_test, y_test))
    print('acc : ', accuracy_score(y_test, y_pred))
