from sklearn.utils import all_estimators
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

path_ddarung = './_data/ddarung/'
path_kaggle = './_data/kaggle_bike/'

ddarung_train = pd.read_csv(path_ddarung + 'train.csv', index_col=0).dropna()
kaggle_train = pd.read_csv(path_kaggle + 'train.csv', index_col=0).dropna()

data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes, ddarung_train, kaggle_train]

algorithms_classifier = all_estimators(type_filter='classifier')
algorithms_regressor = all_estimators(type_filter='regressor')

max_score=0
max_name=''

for i in range(len(data_list)):
    if i<4:
        x, y = data_list[i](return_X_y=True)
        for name, algorithm in algorithms_classifier:
            try:
                model = algorithm()
                model.fit(x, y)
                results = model.score(x, y)
                if max_score<results:
                    max_score=results
                    max_name=name
                print(data_list[i].__name__, name, results)
            except:
                print(data_list[i].__name__, name, 'set default value first')
        print('\n', data_list[i].__name__, 'max_score :', max_name, max_score, '\n')
    elif 4<=i<6:
        x, y = data_list[i](return_X_y=True)
        for name, algorithm in algorithms_regressor:
            try:
                model = algorithm()
                model.fit(x, y)
                results = model.score(x, y)
                if max_score<results:
                    max_score=results
                    max_name=name
                print(data_list[i].__name__, name, results)
            except:
                print(data_list[i].__name__, name, 'set default value first')
        print('\n', data_list[i].__name__, 'max_score :', max_name, max_score, '\n')
    elif i==6:
        x = data_list[i].drop(['count'], axis=1)
        y = data_list[i]['count']
        for name, algorithm in algorithms_regressor:
            try:
                model = algorithm()
                model.fit(x, y)
                results = model.score(x, y)
                if max_score<results:
                    max_score=results
                    max_name=name
                print('ddarung', name, results)
            except:
                print('ddarung', name, 'set deault value first')
        print('\n', 'ddarung max_score :', max_name,  max_score, '\n')
        
    else:
        x = data_list[i].drop(['casual', 'registered', 'count'], axis=1)
        y = data_list[i]['count']
        for name, algorithm in algorithms_regressor:
            try:
                model = algorithm()
                model.fit(x, y)
                results = model.score(x, y)
                if max_score<results:
                    max_score=results
                    max_name=name
                print('kaggle', name, results)
            except:
                print('kaggle', name, 'set deault value first')
        print('\n', 'kaggle max_score :', max_name, max_score), '\n'
                