
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import numpy as np

def RunModel(a, b, c, d, e, f):
    model.fit(a, c)
    g = model.score(b, d)
    print(f, type(e).__name__, 'result : ', g)
    
data_list = [load_iris, load_breast_cancer, load_wine, load_digits, fetch_california_housing, load_diabetes]
classifier_model_list = [RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier, XGBClassifier]
regressor_model_list = [RandomForestRegressor(), DecisionTreeRegressor(), GradientBoostingRegressor(), XGBRegressor()]

for i in range(len(data_list)):
    x, y = data_list[i](return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=337)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    if i<4:
        for j in range(len(classifier_model_list)):
            model = classifier_model_list[j]()
            RunModel(x_train, x_test, y_train, y_test, model, data_list[i].__name__)
            print(data_list[i].__name__, type(model).__name__, 'acc : ', accuracy_score(y_test, model.predict(x_test)))
            fi = model.feature_importances_
            x_fi_train, x_fi_test = x_train, x_test
            index_k = []
            for k in range(len(fi)):
                if fi[k]<np.percentile(fi, 25):
                    index_k.append(k)
            x_fi_train, x_fi_test = pd.DataFrame(x_fi_train).drop(index_k, axis=1), pd.DataFrame(x_fi_test).drop(index_k, axis=1)
            RunModel(x_fi_train, x_fi_test, y_train, y_test, model, data_list[i].__name__)
            print(data_list[i].__name__, type(model).__name__, 'acc : ', accuracy_score(y_test, model.predict(x_fi_test)))

    if 4<=i:
        for j in range(len(regressor_model_list)):
            model = regressor_model_list[j]
            RunModel(x_train, x_test ,y_train, y_test, model, data_list[i].__name__)
            print(data_list[i].__name__, type(model).__name__, 'r2 : ', r2_score(y_test, model.predict(x_test)))
            
            fi = model.feature_importances_
            x_fi_train, x_fi_test = x_train, x_test
            index_k = []
            for k in range(len(fi)):
                if fi[k]<np.percentile(fi,25):
                    index_k.append(k)
            x_fi_train, x_fi_test = pd.DataFrame(x_fi_train).drop(index_k, axis=1), pd.DataFrame(x_fi_test).drop(index_k, axis=1)
            RunModel(x_fi_train, x_fi_test, y_train, y_test, model, data_list[i].__name__)
            print(data_list[i].__name__, type(model).__name__, 'r2 : ', r2_score(y_test, model.predict(x_fi_test)))
            