from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC
import pandas as pd

path_d = './_data/ddarung/'
path_k = './_data/kaggle_bike/'

ddarung = pd.read_csv(path_d + 'train.csv', index_col=0).dropna()
kaggle = pd.read_csv(path_k + 'train.csv', index_col=0).dropna()

data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes, ddarung, kaggle]
scaler_list = [MinMaxScaler(), StandardScaler(), MaxAbsScaler(), RobustScaler()]
classifier_list = [SVC(), RandomForestClassifier(), DecisionTreeClassifier()]
regressor_list = [RandomForestRegressor(), DecisionTreeRegressor()]
cv_list = [GridSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV]

param_svc = [
    {'svc__C':[1, 10, 100]}
]
param_rf = [
    {'randomforestclassifier__n_estimators':[100, 200], 'randomforestclassifier__max_depth':[6, 8, 10], 'randomforestclassifier__min_samples_leaf':[1, 10]}, 
    {'randomforestclassifier__max_depth':[6, 8, 10, 12], 'randomforestclassifier__min_samples_leaf':[3, 5, 7, 10]},
    {'randomforestclassifier__min_samples_leaf':[3, 5, 7, 10], 'randomforestclassifier__min_samples_split':[2, 3, 5, 10]},
    {'randomforestclassifier__min_samples_split':[2, 3, 5, 10]},
    {'randomforestclassifier__n_jobs':[-1, 2, 4], 'randomforestclassifier__min_samples_split':[2, 3, 5, 10]}
]

param_dt = [
    {'decisiontreeclassifier__min_samples_split':[2, 3, 5,10]}
]

for i in range(len(data_list)):
    if i<4:
        x, y = data_list[i](return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
        max_acc = 0
        max_model = ''
        max_scaler = ''
        for j in scaler_list:
            for k in range(len(classifier_list)):
                pipe = make_pipeline(j, classifier_list[k])
                param=[]
                if k == 0:
                    param = param_svc
                elif k == 1:
                    param = param_rf
                elif k == 2:
                    param = param_dt
                for l in cv_list:
                    model = l(pipe, param, cv=5)
                    model.fit(x_train, y_train)
                    if max_acc < accuracy_score(y_test, model.predict(x_test)):
                        max_acc = accuracy_score(y_test, model.predict(x_test))
                        max_model = type(k).__name__
                        max_scaler = type(j).__name__
                        max_cv = l.__name__
                    print('\n', data_list[i].__name__, type(classifier_list[k]).__name__, type(j).__name__, l.__name__, '\nresult : ', model.score(x_test, y_test), '\nacc : ', accuracy_score(y_test, model.predict(x_test)))
        print('\n', data_list[i].__name__, '\nmax_cv : ', max_cv, '\nmax_model : ', max_model, '\nmax_scaler : ', max_scaler, '\nmax_acc : ', max_acc)
    elif 4<=i<6:
        x, y = data_list[i](return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
        max_r2 = 0
        max_model = ''
        max_scaler = ''
        for j in scaler_list:
            for k in range(len(regressor_list)):
                pipe = make_pipeline(j, regressor_list[k])
                param=[]
                if k == 0:
                    param = param_svc
                elif k == 1:
                    param = param_rf
                elif k == 2:
                    param = param_dt
                for l in cv_list:
                    model = l(pipe, param, cv=5)
                    model.fit(x_train, y_train)
                    if max_acc < accuracy_score(y_test, model.predict(x_test)):
                        max_acc = accuracy_score(y_test, model.predict(x_test))
                        max_model = type(k).__name__
                        max_scaler = type(j).__name__
                    print('\n', data_list[i].__name__, type(classifier_list[k]).__name__, type(j).__name__, l.__name__, '\nresult : ', model.score(x_test, y_test), '\nacc : ', accuracy_score(y_test, model.predict(x_test)))
        print('\n', data_list[i].__name__, '\nmax_cv : ', max_cv, '\nmax_model : ', max_model, '\nmax_scaler : ', max_scaler, '\nmax_r2 : ', max_r2)
    elif i==6:
        x = data_list[i].drop(['count'], axis=1)
        y = data_list[i]['count']
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
        max_r2 = 0
        max_model = ''
        max_scaler = ''
        for j in scaler_list:
            for k in range(len(regressor_list)):
                pipe = make_pipeline(j, regressor_list[k])
                param=[]
                if k == 0:
                    param = param_svc
                elif k == 1:
                    param = param_rf
                elif k == 2:
                    param = param_dt
                for l in cv_list:
                    model = l(pipe, param, cv=5)
                    model.fit(x_train, y_train)
                    if max_acc < accuracy_score(y_test, model.predict(x_test)):
                        max_acc = accuracy_score(y_test, model.predict(x_test))
                        max_model = type(k).__name__
                        max_scaler = type(j).__name__
                    print('\n', data_list[i].__name__, type(classifier_list[k]).__name__, type(j).__name__, l.__name__, '\nresult : ', model.score(x_test, y_test), '\nacc : ', accuracy_score(y_test, model.predict(x_test)))
        print('\n ddarung  \nmax_cv : ', max_cv, '\nmax_model : ', max_model, '\nmax_scaler : ', max_scaler, '\nmax_r2 : ', max_r2)
    elif i==7:
        x = data_list[i].drop(['casual', 'registered', 'count'], axis=1)
        y = data_list[i]['count']
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
        max_r2 = 0
        max_model = ''
        max_scaler = ''
        for j in scaler_list:
            for k in range(len(regressor_list)):
                pipe = make_pipeline(j, regressor_list[k])
                param=[]
                if k == 0:
                    param = param_svc
                elif k == 1:
                    param = param_rf
                elif k == 2:
                    param = param_dt
                for l in cv_list:
                    model = l(pipe, param, cv=5)
                    model.fit(x_train, y_train)
                    if max_acc < accuracy_score(y_test, model.predict(x_test)):
                        max_acc = accuracy_score(y_test, model.predict(x_test))
                        max_model = type(k).__name__
                        max_scaler = type(j).__name__
                    print('\n', data_list[i].__name__, type(classifier_list[k]).__name__, type(j).__name__, l.__name__, '\nresult : ', model.score(x_test, y_test), '\nacc : ', accuracy_score(y_test, model.predict(x_test)))
        print('\n kaggle  \nmax_cv : ', max_cv, '\nmax_model : ', max_model, '\nmax_scaler : ', max_scaler, '\nmax_r2 : ', max_r2)