# 회귀 데이터 싹 다 모아서 테스트
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import  RandomForestRegressor, RandomForestClassifier
import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score

# 1. 데이터
path_ddarung = './_data/ddarung/'
path_kaggle_bike = './_data/kaggle_bike/'

ddarung_train = pd.read_csv(path_ddarung + 'train.csv', index_col=0).dropna()
ddarung_test = pd.read_csv(path_ddarung + 'test.csv', index_col=0).dropna()
kaggle_train = pd.read_csv(path_kaggle_bike + 'train.csv', index_col=0)
kaggle_test = pd.read_csv(path_kaggle_bike + 'test.csv', index_col=0)

data_list = [fetch_california_housing, load_diabetes, ddarung_train, kaggle_train]
name_list = ['fetch_california_housing', 'load_diabetes', 'ddarung_train', 'kaggle_train']

model_list = [DecisionTreeRegressor(), RandomForestRegressor()]

for i in range(len(data_list)):
    if i<2:
        x, y = data_list[i](return_X_y=True) 
    elif i==2:
        x = ddarung_train.drop(['count'], axis=1)
        y = ddarung_train['count']
    elif i==3:
        x = kaggle_train.drop(['casual', 'registered', 'count'], axis=1)
        y = kaggle_train['count']
    for j in model_list:
        model = j
        model.fit(x, y)
        results = model.score(x, y)
        print(name_list[i], type(j).__name__, results)
        
