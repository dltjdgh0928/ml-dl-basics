# [실습] 각종 모델 넣기

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, train_size=0.8, shuffle=True, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
aaa = LogisticRegression()
model = BaggingClassifier(aaa, n_estimators=10, n_jobs=-1, random_state=337, bootstrap=True)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# 4. 평가, 예측
print('score : ', model.score(x_test, y_test))
print('acc : ', accuracy_score(y_test, y_pred))
