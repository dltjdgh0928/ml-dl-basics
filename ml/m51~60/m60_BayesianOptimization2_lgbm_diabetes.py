from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
import numpy as np

scaler = MinMaxScaler()

bayesian_params = {
    'max_depth' : (3, 16),
    'num_leaves' : (24, 64),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (10, 500),
    'reg_lambda' : (0.001, 10),
    'reg_alpha' : (0.01, 50)
}

x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


def LGB(max_depth, num_leaves, min_child_samples, min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    
    model = LGBMRegressor(max_depth=int(max_depth),
                          num_leaves=int(num_leaves),
                          min_child_samples=int(min_child_samples),
                          min_child_weight=int(min_child_weight),
                          subsample=subsample,
                          colsample_bytree=colsample_bytree,
                          max_bin=int(max_bin),
                          reg_lambda=reg_lambda,
                          reg_alpha=reg_alpha)
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    return r2


bo = BayesianOptimization(
    f=LGB,
    pbounds=bayesian_params,
    random_state=123    
)

bo.maximize(init_points=2, n_iter=20)
print(bo.max)