from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits, fetch_covtype, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import warnings
warnings.filterwarnings(action='ignore')

data_list = [load_iris(return_X_y=True), load_breast_cancer(return_X_y=True), load_wine(return_X_y=True)]

model_list = [LinearSVC(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]

data_name_list = ['iris : ', 'breast_cancer : ', 'wine : ']

model_name_list = ['LinearSVC : ', 'LogisticRegression :', 'DecisionTreeClassifier : ', 'RandomForestClassifier : ']

for i, v in enumerate(data_list):
    x, y = v
    print(data_name_list[i])
    for j, v in enumerate(model_list):
        model = v
        model.fit(x, y)
        results = model.score(x, y)
        print(model_name_list[j], results)