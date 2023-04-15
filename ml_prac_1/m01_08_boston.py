from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing, load_diabetes
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

path_ddarung = './_data/ddarung/'
path_kaggle = './_data/kaggle_bike/'

ddarung_train = pd.read_csv(path_ddarung + 'train.csv', index_col=0).dropna()
kaggle_train = pd.read_csv(path_kaggle + 'train.csv', index_col=0).dropna()

data_list = [fetch_california_housing, load_diabetes, ddarung_train, kaggle_train]
name_list = ['fetch_california_housing', 'load_diabetes', 'ddarung_train', 'kaggle_train']
model_list = [DecisionTreeRegressor(), RandomForestRegressor()]

for i in range(len(data_list)):
    if i<2:
        x, y = data_list[i](return_X_y=True)
    elif i==2:
        x = data_list[i].drop(['count'], axis=1)
        y = data_list[i]['count']
    else:
        x = data_list[i].drop(['casual', 'registered', 'count'], axis=1)
        y = data_list[i]['count']
    for j in model_list:
        model = j
        model.fit(x, y)
        results = model.score(x, y)
        print(name_list[i], type(j).__name__, results)