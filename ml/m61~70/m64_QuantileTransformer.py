from sklearn.datasets import fetch_california_housing, load_diabetes, load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
data_list = [fetch_california_housing, load_diabetes, load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype]
scaler_list = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), QuantileTransformer(n_quantiles=1000), QuantileTransformer(n_quantiles=1000), PowerTransformer(method='yeo-johnson')
            #    , PowerTransformer(method='box-cox')
               ]
reg_model_list = [LinearRegression, RandomForestRegressor]
cla_model_list = [LogisticRegression, RandomForestClassifier]

for i in range(len(data_list)):
        
    x, y = data_list[i](return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, train_size=0.8, random_state=337
    )
    for j in scaler_list:
        scaler = j
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # 2. 모델
        if i<2:
            for k in range(len(reg_model_list)):
                model = reg_model_list[k]()

                # 3. 훈련
                model.fit(x_train, y_train)

                # 4. 평가, 예측
                print(data_list[i].__name__, j, reg_model_list[k], 'result : ', round(model.score(x_test, y_test), 2))
        elif i>=2:
            for l in range(len(cla_model_list)):
                model = cla_model_list[l]()

                # 3. 훈련
                model.fit(x_train, y_train)

                # 4. 평가, 예측
                print(data_list[i].__name__, j, cla_model_list[l], 'result : ', round(model.score(x_test, y_test), 2))
            