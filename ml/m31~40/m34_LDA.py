# Linear Discriminant Analysis ( 지도 학습임, pca는 비지도 )

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def Runmodel(x, y, name, a):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(a, name.__name__, 'result : ', result)
    y_pred = model.predict(x_test)
    print(a, name.__name__, 'predict acc : ', accuracy_score(y_test, y_pred))
def LDAmodel(x, y, name, a):
    x_lda = lda.fit_transform(x, y)
    Runmodel(x_lda, y, name, a)
def Model():
    Runmodel(x, y, data_list[i], '')
    LDAmodel(x, y, data_list[i], 'LDA')
def dacon_diabetes():
    return None
def dacon_wine():
    return None

path_d = './_data/dacon_diabetes/'
path_w = './_data/wine/'

diabetes = pd.read_csv(path_d + 'train.csv', index_col=0).dropna()
wine = pd.read_csv(path_w + 'train.csv', index_col=0).dropna()

d_x = diabetes.drop(['Outcome'], axis=1)
d_y = diabetes['Outcome']

w_x = wine.drop(['quality'], axis=1)
w_x['type'] = le.fit_transform(w_x['type'])
w_y = wine['quality']

data_list = [load_iris, load_breast_cancer, load_digits, load_wine, dacon_diabetes, dacon_wine]

lda = LinearDiscriminantAnalysis()
model = RandomForestClassifier()

for i in range(len(data_list)):
    if i<3:
        x, y = data_list[i](return_X_y=True)
        Model()
    else:
        Model()