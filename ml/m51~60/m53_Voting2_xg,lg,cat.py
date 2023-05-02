import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, train_size=0.8, shuffle=True)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델

xg = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor()

model = VotingRegressor(
    estimators=[('XG', xg), ('LG', lg), ('CAT', cat)],
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# 4. 평가, 예측
print('score : ', model.score(x_test, y_test))
print('r2 : ', r2_score(y_test, y_pred))

classifiers = [xg, lg, cat]
for model2 in classifiers:
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = r2_score(y_test, y_predict)
    class_name = model2.__class__.__name__
    print('{0} r2 : {1:.4f}'.format(class_name, score2))