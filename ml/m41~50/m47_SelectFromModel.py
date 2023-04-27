import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import pandas as pd
import warnings
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')

# 1. 데이터
data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes]
scaler = RobustScaler()
parameters = {'n_estimators' : 1000,
              'learning_rate' : 0.3,
              'max_depth' : 2,
              'gamma' : 0,
              'min_child_weight' : 0,
              'subsample' : 0.2,
              'colsample_bytree' : 0.5,
              'colsample_bylevel' : 0,
              'colsample_bynode' : 0,
              'reg_alpha' : 0,
              'reg_lambda' : 1,
              'random_state' : 337,
              }
model = XGBClassifier(**parameters)

for i in range(len(data_list)):
    x, y = data_list[i](return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8)
    x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)
    if i<4:
        model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=10, verbose=0, eval_metric='merror')
    else:
        model = XGBRegressor(**parameters)
        model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=10, verbose=0, eval_metric='rmse')
    print('result : ', model.score(x_test, y_test))
        
    thresholds = np.sort(model.feature_importances_)
    for j in thresholds:
        selection = SelectFromModel(model, threshold=j, prefit=True)
        select_x_train, select_x_test = selection.transform(x_train), selection.transform(x_test)
        if i<4:
            selection_model = XGBClassifier()
            selection_model.set_params(early_stopping_rounds=10, **parameters, eval_metric='merror')
        else:
            selection_model = XGBRegressor()
            selection_model.set_params(early_stopping_rounds=10, **parameters, eval_metric='rmse')
        selection_model.fit(select_x_train, y_train, eval_set=[(select_x_train, y_train), (select_x_test, y_test)], verbose=0)
        select_y_predict = selection_model.predict(select_x_test)
        score = model.score(y_test, select_y_predict)
        print('Thres=%.3f, n=%d, R2: %.2f%%' %(i, select_x_train.shape[1], score*100))