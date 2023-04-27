import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import pandas as pd
import warnings
from sklearn.feature_selection import SelectFromModel
warnings.filterwarnings('ignore')
# 1. 데이터
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
model = XGBRegressor(**parameters)
def Runmodel(a, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8)
    x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)
    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=10, verbose=0, eval_metric='rmse')
    print(a, 'result : ', model.score(x_test, y_test))
    y_predict = model.predict(x_test)
    r2 = r2_score(y_test, y_predict)
    print(a, 'r2 : ', r2)
    mse = mean_squared_error(y_test, y_predict)
    print(a, 'rmse : ', np.sqrt(mse))
    
x, y = load_diabetes(return_X_y=True)
# Runmodel('remain all', x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8)
x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)
model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=10, verbose=0, eval_metric='rmse')
print('result : ', model.score(x_test, y_test))
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)
mse = mean_squared_error(y_test, y_predict)
print('rmse : ', np.sqrt(mse))
    
# for i in range(x.shape[1]-1):
#     a = model.feature_importances_
#     b = np.argmin(a, axis=0)
#     x = pd.DataFrame(pd.DataFrame(x).drop(b, axis=1).values)
#     Runmodel(f'remain {9-i} column', x, y)

print(model.feature_importances_)
thresholds = np.sort(model.feature_importances_)

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=True)           # False 면 다시 훈련
    select_x_train, select_x_test = selection.transform(x_train), selection.transform(x_test)
    
    selection_model = XGBRegressor()
    selection_model.set_params(early_stopping_rounds=10, **parameters, eval_metric='rmse')
    selection_model.fit(select_x_train, y_train, eval_set=[(select_x_train, y_train), (select_x_test, y_test)], verbose=0)
    
    select_y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, select_y_predict)
    print('Thres=%.3f, n=%d, R2: %.2f%%' %(i, select_x_train.shape[1], score*100))