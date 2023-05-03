from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
import numpy as np
import time
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
def rmse(x, y):
    return np.sqrt(mean_squared_error(x, y))

data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes]
for i in range(len(data_list)):
    x, y = data_list[i](return_X_y=True)
    if i == 4:
        y=y-1
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)


    # 2. 모델
    parameters = {'learning_rate' : (3, 16),
                'max_depth' : (3, 16),
                'gamma' : (0.1, 1),
                'min_child_weight' : (0, 100),
                'subsample' : (0.5, 0.9),
                'colsample_bytree' : (0.5, 0.9),
                'reg_alpha' : (0.1, 10),
                'reg_lambda' : (0.1, 10)}

    def xgb_hamsu(learning_rate, max_depth, gamma, min_child_weight, subsample,
                colsample_bytree, reg_lambda, reg_alpha):
        params = {
            'n_estimators' : 1000,
            'learning_rate' : learning_rate,
            'max_depth' : int(round(max_depth)),
            'gamma' : int(round(gamma)),
            'min_child_weight' : int(round(min_child_weight)),
            'subsample' : max(min(subsample, 1), 0),
            'colsample_bytree' : colsample_bytree,
            'reg_lambda' : max(reg_lambda, 0),
            'reg_alpha' : max(reg_alpha, 0)
        }
        if i<5:
            model = XGBClassifier(**params)
            model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], eval_metric='auc', verbose=0, early_stopping_rounds=50)
        elif i>=5:
            model = XGBRegressor(**params)
            model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], eval_metric='rmse', verbose=0, early_stopping_rounds=50)
        y_predict = model.predict(x_test)
        if i<5:
            result = accuracy_score(y_test, y_predict)
        elif i>=5:
            result = r2_score(y_test, y_predict)
        return result

    lgb_bo = BayesianOptimization(
        f=xgb_hamsu,
        pbounds=parameters,
        random_state=337
    )

    n_iter=10
    start_time = time.time()
    lgb_bo.maximize(init_points=5, n_iter=n_iter)
    end_time = time.time()
    print(lgb_bo.max)
    print(data_list[i].__name__, n_iter, '번 걸린 시간 : ', end_time-start_time)