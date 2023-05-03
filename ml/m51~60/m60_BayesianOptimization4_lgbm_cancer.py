from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np
import time
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델
bayesian_params = {
    'max_depth' : (3, 16),
    'learning_rate' : (3, 16),
    'num_leaves' : (24, 64),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (10, 500),
    'reg_lambda' : (0.001, 10),
    'reg_alpha' : (0.01, 50)
}

def rmse(x, y):
    return np.sqrt(mean_squared_error(x, y))

def lgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight, subsample,
              colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 1000,
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample, 1), 0),
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)), 10),
        'reg_lambda' : max(reg_lambda, 0),
        'reg_alpha' : reg_alpha
    }
    
    model = LGBMClassifier(**params)
    
    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='rmse', verbose=0, early_stopping_rounds=50)
    y_predict = model.predict(x_test)
    
    result = -rmse(y_test, y_predict)
    return result

lgb_bo = BayesianOptimization(
    f=lgb_hamsu,
    pbounds=bayesian_params,
    random_state=337
)

n_iter=100
start_time = time.time()
lgb_bo.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()
print(lgb_bo.max)
print(n_iter, '번 걸린 시간 : ', end_time-start_time)