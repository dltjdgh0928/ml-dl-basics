import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

def fit(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    return result
    
datasets = load_breast_cancer()
x=datasets['data']
y=datasets.target

result = fit(x,y)
print('result : ', result)

for i in range(10):
    pca = PCA(n_components=10-i)
    x = pca.fit_transform(x)
    current_result = fit(x, y)
    print(f'reduce {i} dimensions result : ', current_result)