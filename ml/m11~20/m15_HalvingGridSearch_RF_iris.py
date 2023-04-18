import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import accuracy_score, r2_score
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
path_d = './_data/ddarung/'
path_k = './_data/kaggle_bike/'
ddarung = pd.read_csv(path_d + 'train.csv', index_col=0).dropna()
kaggle = pd.read_csv(path_k + 'train.csv', index_col=0).dropna()

data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes, ddarung, kaggle]

parameter = [
    {'n_estimators':[100, 200], 'max_depth':[6,8,10], 'min_samples_leaf':[1,10]}, 
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
    {'min_samples_split':[2,3,5,10]},
    {'n_jobs':[-1, 2, 4], 'min_samples_split':[2,3,5,10]}
]

n_splits=10
stkf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
scaler = RobustScaler()

for i in range(len(data_list)):
    if i<5:
        x, y = data_list[i](return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = HalvingGridSearchCV(RandomForestClassifier(), parameter, cv=stkf, refit=True)
        st = time.time()
        model.fit(x_train, y_train)
        et = time.time()
        print(data_list[i].__name__, 'time : ', et-st, 's')
        print(data_list[i].__name__, 'y_pred acc : ', accuracy_score(y_test, model.predict(x_test)))
        print(data_list[i].__name__, 'best estimator : ', model.best_estimator_)
        print(data_list[i].__name__, 'best params : ', model.best_params_)
        print(data_list[i].__name__, 'best score : ', model.best_score_)
        print(data_list[i].__name__, 'y_pred_best acc : ', accuracy_score(y_test, model.best_estimator_.predict(x_test)))
    if 5<=i<7:
        x, y = data_list[i](return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = HalvingGridSearchCV(RandomForestRegressor(), parameter, cv=kf, refit=True)
        st = time.time()
        model.fit(x_train, y_train)
        et = time.time()
        print(data_list[i].__name__, 'time : ', et-st, 's')
        print(data_list[i].__name__, 'y_pred r2 : ', r2_score(y_test, model.predict(x_test)))
        print(data_list[i].__name__, 'best estimator : ', model.best_estimator_)
        print(data_list[i].__name__, 'best params : ', model.best_params_)
        print(data_list[i].__name__, 'best score : ', model.best_score_)
        print(data_list[i].__name__, 'y_pred_best r2 : ', r2_score(y_test, model.best_estimator_.predict(x_test)))
    elif i==7:
        x_train = ddarung.drop(['count'], axis=1)
        y_train = ddarung['count']
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = HalvingGridSearchCV(RandomForestRegressor(), parameter, cv=kf, refit=True)
        st = time.time()
        model.fit(x_train, y_train)
        et = time.time()
        print('ddarung time : ', et-st, 's')
        print('ddarung y_pred r2 : ', r2_score(y_test, model.predict(x_test)))
        print('ddarung best estimator : ', model.best_estimator_)
        print('ddarung best params : ', model.best_params_)
        print('ddarung best score : ', model.best_score_)
        print('ddarung y_pred_best r2 : ', r2_score(y_test, model.best_estimator_.predict(x_test)))
    elif i==8:
        x = kaggle.drop(['casual', 'registered', 'count'], axis=1)
        y = kaggle['count']
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = HalvingGridSearchCV(RandomForestRegressor(), parameter, cv=kf, refit=True)
        st = time.time()
        model.fit(x_train, y_train)
        et = time.time()
        print('kaggle time : ', et-st, 's')
        print('kaggle y_pred r2 : ', r2_score(y_test, model.predict(x_test)))
        print('kaggle best estimator : ', model.best_estimator_)
        print('kaggle best params : ', model.best_params_)
        print('kaggle best score : ', model.best_score_)
        print('kaggle y_pred_best r2 : ', r2_score(y_test, model.best_estimator_.predict(x_test)))
        
