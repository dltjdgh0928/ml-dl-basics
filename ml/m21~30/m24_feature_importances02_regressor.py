import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.utils import to_categorical

# 1. 데이터
data_list = [fetch_california_housing, load_diabetes]

model_list = [DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, XGBRegressor]

for i in data_list:
    x, y = i(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=337)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 2. 모델
    for j in model_list:
        model = j()
        # 3. 훈련
        model.fit(x_train, y_train)

        # 4. 평가, 예측
        result = model.score(x_test, y_test)
        print(i.__name__, j.__name__, 'model.score : ', result)

        y_pred = model.predict(x_test)
        acc = r2_score(y_test, y_pred)
        print(i.__name__, j.__name__, 'acc : ', acc)

        print(i.__name__, j.__name__, ':', model.feature_importances_)
    
# DecisionTreeClassifier() : [0.         0.01671193 0.93062443 0.05266364]
# RandomForestClassifier() : [0.13938151 0.03362868 0.41336426 0.41362556]
# GradientBoostingClassifier() : [0.0061074  0.01343872 0.72144525 0.25900863]
# XGBClassifier() : [0.01794496 0.01218657 0.8486943  0.12117416]