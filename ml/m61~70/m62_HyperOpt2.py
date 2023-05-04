from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
import numpy as np
import time
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# 2. 모델
search_space = {
    'learning_rate' : hp.uniform('learning_rate', 0, 0.001),
    'max_depth' : hp.quniform('max_depth', 3, 16, 1),
    'num_leaves' : hp.quniform('num_leaves', 24, 64, 1),
    'min_child_samples' : hp.quniform('min_child_samples', 10, 200, 1),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 50, 1),
    'subsample' : hp.uniform('subsample', 0.5, 1),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
    'max_bin' : hp.quniform('max_bin', 10, 500, 1),
    'reg_lambda' : hp.uniform('reg_lambda', 0.001, 10),
    'reg_alpha' : hp.uniform('reg_alpha', 0.01, 50)
}
# hp.quniform(label, low, high, q) : 최소부터 최대까지 q 간격
# hp.uniform(lable, low, high) : 최소부터 최대 정규분포 간격
# hp.randint(label, upper) : 0부터 최대값 upper까지 random한 정수값
# hp.loguniform(label, upper) : exp(uniform[low, high))값 반환 / 이거 역시 정규분포



def lgb_hamsu(search_space):
    params = {
        'n_estimators' : 1000,
        'learning_rate' : search_space['learning_rate'],
        'max_depth' : int(search_space['max_depth']),
        'num_leaves' : int(search_space['num_leaves']),
        'min_child_samples' : int(search_space['min_child_samples']),
        'min_child_weight' : int(search_space['min_child_weight']),
        'subsample' : search_space['subsample'],
        'colsample_bytree' : search_space['colsample_bytree'],
        'max_bin' : int(search_space['max_bin']),
        'reg_lambda' : search_space['reg_lambda'],
        'reg_alpha' : search_space['reg_alpha']
    }
    
    model = LGBMRegressor(**params)
    
    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='rmse', verbose=0, early_stopping_rounds=50)
    y_predict = model.predict(x_test)
    
    result = mean_squared_error(y_test, y_predict)
    return result

trial_val = Trials()

best = fmin(
    fn=lgb_hamsu,
    space=search_space,
    algo = tpe.suggest,
    max_evals=50,
    trials=trial_val,
    rstate=np.random.default_rng(seed=10)
)

print('best : ', best)


results = [i['loss'] for i in trial_val.results]
import pandas as pd
df = pd.DataFrame({'learning_rate' : trial_val.vals['learning_rate'],
                   'max_depth' : trial_val.vals['max_depth'],
                   'num_leaves' : trial_val.vals['num_leaves'],
                   'min_child_samples' : trial_val.vals['min_child_samples'],
                   'min_child_weight' : trial_val.vals['min_child_weight'],
                   'subsample' : trial_val.vals['subsample'],
                   'colsample_bytree' : trial_val.vals['colsample_bytree'],
                   'max_bin' : trial_val.vals['max_bin'],
                   'reg_lambda' : trial_val.vals['reg_lambda'],
                   'reg_alpha' : trial_val.vals['reg_alpha'],
                   'results' : results})

print(df)