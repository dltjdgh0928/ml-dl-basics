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
        model = HalvingRandomSearchCV(RandomForestClassifier(), parameter, cv=stkf, refit=True)
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
        model = HalvingRandomSearchCV(RandomForestRegressor(), parameter, cv=kf, refit=True)
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
        model = HalvingRandomSearchCV(RandomForestRegressor(), parameter, cv=kf, refit=True)
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
        model = HalvingRandomSearchCV(RandomForestRegressor(), parameter, cv=kf, refit=True)
        st = time.time()
        model.fit(x_train, y_train)
        et = time.time()
        print('kaggle time : ', et-st, 's')
        print('kaggle y_pred r2 : ', r2_score(y_test, model.predict(x_test)))
        print('kaggle best estimator : ', model.best_estimator_)
        print('kaggle best params : ', model.best_params_)
        print('kaggle best score : ', model.best_score_)
        print('kaggle y_pred_best r2 : ', r2_score(y_test, model.best_estimator_.predict(x_test)))
        

# load_iris time :  1.5225186347961426 s
# load_iris y_pred acc :  0.9555555555555556
# load_iris best estimator :  RandomForestClassifier(min_samples_leaf=3)
# load_iris best params :  {'min_samples_split': 2, 'min_samples_leaf': 3}
# load_iris best score :  0.9266666666666667
# load_breast_cancer time :  19.50419330596924 s
# load_breast_cancer y_pred acc :  0.9824561403508771
# load_breast_cancer best estimator :  RandomForestClassifier(max_depth=10)
# load_breast_cancer best params :  {'n_estimators': 100, 'min_samples_leaf': 1, 'max_depth': 10}
# load_breast_cancer best score :  0.9551587301587302
# load_breast_cancer y_pred_best acc :  0.9824561403508771
# load_digits time :  15.895793199539185 s
# load_digits y_pred acc :  0.9685185185185186
# load_digits best estimator :  RandomForestClassifier(min_samples_split=5, n_jobs=-1)
# load_digits best params :  {'n_jobs': -1, 'min_samples_split': 5}
# load_digits best score :  0.9548022598870055
# load_digits y_pred_best acc :  0.9685185185185186
# load_wine time :  3.6362478733062744 s
# load_wine y_pred acc :  0.9259259259259259
# load_wine best estimator :  RandomForestClassifier(min_samples_leaf=3, min_samples_split=3)
# load_wine best params :  {'min_samples_split': 3, 'min_samples_leaf': 3}
# load_wine best score :  0.9833333333333334
# load_wine y_pred_best acc :  0.9259259259259259
# C:\Users\bitcamp\anaconda3\envs\tf273cpu\lib\site-packages\sklearn\model_selection\_search.py:292: UserWarning: The total space of parameters 60 is smaller than n_iter=2905. Running 60 iterations. For exhaustive searches, use GridSearchCV.  warnings.warn(
# PS C:\study\study>  c:; cd 'c:\study\study'; & 'C:\Users\bitcamp\anaconda3\envs\tf273cpu\python.exe' 'c:\Users\bitcamp\.vscode\extensions\ms-python.python-2023.6.0\pythonFiles\lib\python\debugpy\adapter/../..\debugpy\launcher' '61454' '--' 'c:\study\study\ml\m11~20\m16_HalvingRandomSearch_RF_iris.py'
# load_iris time :  2.569509267807007 s
# load_iris y_pred acc :  0.9333333333333333
# load_iris best estimator :  RandomForestClassifier(max_depth=10, n_estimators=200)
# load_iris best params :  {'n_estimators': 200, 'min_samples_leaf': 1, 'max_depth': 10}
# load_iris best score :  0.9666666666666668
# load_iris y_pred_best acc :  0.9333333333333333
# load_breast_cancer time :  21.312559366226196 s
# load_breast_cancer y_pred acc :  0.9824561403508771
# load_breast_cancer best estimator :  RandomForestClassifier(max_depth=8, n_estimators=200)
# load_breast_cancer best params :  {'n_estimators': 200, 'min_samples_leaf': 1, 'max_depth': 8}
# load_breast_cancer best score :  0.9580158730158731
# load_breast_cancer y_pred_best acc :  0.9824561403508771
# load_digits time :  18.034666299819946 s
# load_digits y_pred acc :  0.9722222222222222
# load_digits best estimator :  RandomForestClassifier(max_depth=10, n_estimators=200)
# load_digits best params :  {'n_estimators': 200, 'min_samples_leaf': 1, 'max_depth': 10}
# load_digits best score :  0.943135593220339
# load_digits y_pred_best acc :  0.9722222222222222
# load_wine time :  2.4919703006744385 s
# load_wine y_pred acc :  1.0
# load_wine best estimator :  RandomForestClassifier(min_samples_split=5)
# load_wine best params :  {'min_samples_split': 5}
# load_wine best score :  0.9666666666666668
# load_wine y_pred_best acc :  1.0
# fetch_covtype time :  329.4170138835907 s
# fetch_covtype y_pred acc :  0.9488652010280889
# fetch_covtype best estimator :  RandomForestClassifier(min_samples_split=5)
# fetch_covtype best params :  {'min_samples_split': 5}
# fetch_covtype best score :  0.7599160737091772
# fetch_covtype y_pred_best acc :  0.9488652010280889
# fetch_california_housing time :  128.92535710334778 s
# fetch_california_housing y_pred r2 :  0.8104695796392999
# fetch_california_housing best estimator :  RandomForestRegressor(min_samples_split=10, n_jobs=4)
# fetch_california_housing best params :  {'n_jobs': 4, 'min_samples_split': 10}
# fetch_california_housing best score :  0.7037482201600137
# fetch_california_housing y_pred_best r2 :  0.8104695796392999
# load_diabetes time :  31.562979221343994 s
# load_diabetes y_pred r2 :  0.4486849354005267
# load_diabetes best estimator :  RandomForestRegressor(min_samples_leaf=5, min_samples_split=10)
# load_diabetes best params :  {'min_samples_split': 10, 'min_samples_leaf': 5}
# load_diabetes best score :  0.34033880838055286
# load_diabetes y_pred_best r2 :  0.4486849354005267
# ddarung time :  22.500677585601807 s
# ddarung y_pred r2 :  0.4626155041517598
# ddarung best estimator :  RandomForestRegressor(max_depth=10, min_samples_leaf=10)
# ddarung best params :  {'n_estimators': 100, 'min_samples_leaf': 10, 'max_depth': 10}
# ddarung best score :  0.3965134118599617
# ddarung y_pred_best r2 :  0.4626155041517598
# kaggle time :  122.68241882324219 s
# kaggle y_pred r2 :  0.3390314982169331
# kaggle best estimator :  RandomForestRegressor(max_depth=8, min_samples_leaf=7)
# kaggle best params :  {'min_samples_leaf': 7, 'max_depth': 8}
# kaggle best score :  0.23067161546584844
# kaggle y_pred_best r2 :  0.3390314982169331
