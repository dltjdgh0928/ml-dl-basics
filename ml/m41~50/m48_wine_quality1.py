import pandas as pd
import numpy as np
import optuna
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import datetime
import warnings
warnings.filterwarnings('ignore')
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')

le = LabelEncoder()
imputer = IterativeImputer(XGBRegressor())
scaler = MinMaxScaler()
path = './_data/wine/'
path_save = './_save/wine/'

wine = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample = pd.read_csv(path + 'sample_submission.csv', index_col=0)
wine['type'] = le.fit_transform(wine['type'])
test_csv['type'] = le.fit_transform(test_csv['type'])
test_csv = imputer.fit_transform(test_csv)
x = imputer.fit_transform(wine.drop(['quality'], axis=1))
y = wine['quality'] - 3
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    learning_rate  = trial.suggest_loguniform('learning_rate', 0.001, 1)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    gamma = trial.suggest_int('gamma', 1, 100)
    min_child_weight = trial.suggest_loguniform('min_child_weight', 0.1, 100)
    subsample = trial.suggest_loguniform('subsample', 0.1, 1)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0, 1)
    colsample_bylevel = trial.suggest_float('colsample_bylevel', 0, 1)
    colsample_bynode = trial.suggest_float('colsample_bynode', 0, 1)
    reg_alpha = trial.suggest_loguniform('reg_alpha', 0.001, 10)
    reg_lambda = trial.suggest_loguniform('reg_lambda', 0.001, 10)

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        gamma=gamma,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=colsample_bylevel,
        colsample_bynode=colsample_bynode,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda
        )
    model.fit(x_train, y_train)
    print('result : ', model.score(x_test, y_test))
    y_pred = np.round(model.predict(x_test))
    acc = accuracy_score(y_test, y_pred)
    print('test acc : ', acc)
    if 0.55<=acc:
        sample['quality'] = model.predict(test_csv) + 3
        sample.to_csv(path_save + 'wine_' + date + '.csv')
    return acc
        
opt = optuna.create_study(direction='maximize')
opt.optimize(objective, n_trials=100)
print('best param : ', opt.best_params, 'best rmse : ', opt.best_value)


