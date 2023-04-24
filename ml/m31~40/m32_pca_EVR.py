import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

datasets = load_breast_cancer()
x=datasets['data']
y=datasets.target

pca = PCA(n_components=30)
x = pca.fit_transform(x)
print(x.shape)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR))     # 0.9999999999999998

pca_cumsum = np.cumsum(pca_EVR)
print(pca_cumsum)

import matplotlib.pyplot as plt
plt.plot(pca_cumsum)
plt.grid()
plt.show()










# model = RandomForestRegressor()

# def fit(x, y):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)
#     model.fit(x_train, y_train)
#     result = model.score(x_test, y_test)
#     return result
    
# datasets = load_breast_cancer()
# x=datasets['data']
# y=datasets.target
# print(x.shape)
# result = fit(x,y)
# print('result : ', result)

# for i in range(x.shape[1]):
#     pca = PCA(n_components=x.shape[1]-i)
#     x_p = pca.fit_transform(x)
#     current_result = fit(x_p, y)
#     print(f'reduce {i} dimensions result : ', current_result)