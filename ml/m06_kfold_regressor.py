from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

path_ddarung = './_data/ddarung/'
path_kaggle = './_data/kaggle_bike/'

data_list = [load_boston, fetch_california_housing, ]
n_split = 10
kf = KFold(n_splits=n_split, shuffle=True, random_state=123)
for i in data_list:
    x, y = i(return_X_y=True)
    scores = cross_val_score(RandomForestClassifier(), x, y, cv=kf)
    print('\n', i.__name__, '\nacc : ', scores, '\n mean of cross_val_score : ', round(np.mean(scores)))
    