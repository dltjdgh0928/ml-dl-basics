import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model = RandomForestClassifier()

def fit(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    return result
    
data_list = [load_iris, load_breast_cancer, load_wine, load_digits, fetch_california_housing, load_diabetes]

for i in range(len(data_list)):
    x, y = data_list[i](return_X_y=True)
    if i<4:
        result = fit(x,y)
        print('result : ', result)
        for i in range(x.shape[1]):
            pca = PCA(n_components=x.shape[1]-i)
            x_p = pca.fit_transform(x)
            current_result = fit(x_p, y)
            print(f'reduce {i} dimensions result : ', current_result)
    elif 4<=i:
        model = RandomForestRegressor()
        result = fit(x,y)
        print('result : ', result)
        for i in range(x.shape[1]):
            pca = PCA(n_components=x.shape[1]-i)
            x_p = pca.fit_transform(x)
            current_result = fit(x_p, y)
            print(f'reduce {i} dimensions result : ', current_result)