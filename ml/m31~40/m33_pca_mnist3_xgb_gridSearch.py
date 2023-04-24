# n_component > 0.95 이상
# xgboost, gridSearch 또는 RandomSearch를 쓸 것

# m33_2 결과 뛰어넘기

parameters = [
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.3, 0.001, 0.01],
     'max_depth':[4, 5, 6]},
    {'n_estimators':[90, 100, 110], 'learning_rate':[0.1, 0.001, 0.01],
     'max_depth':[4, 5, 6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators':[90, 110], 'learning_rate':[0.1, 0.001, 0.01],
     'max_depth':[4, 5, 6], 'colsample_bytree':[0.6, 0.9, 1],
     'colsample_bylevel':[0.6, 0.7, 0.9]},
]

# n_jobs = -1
#     tree_method = 'gpu_hist'
#     predictor = 'gpu_predictor'
#     gpu_id = 0

import numpy as np
from xgboost import XGBClassifier
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

n_c_list = [154, 331, 486, 713]
pca_list = [0.95, 0.99, 0.999, 1.0]

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)
y = to_categorical(y)
x = x.reshape(x.shape[0], -1)
