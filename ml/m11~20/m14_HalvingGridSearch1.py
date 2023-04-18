import numpy as np
from sklearn.datasets import load_digits
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time
import pandas as pd

# 1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=12345, test_size=0.2)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=337)
# kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)

parameters = [
    {'C':[1,10,100,1000], 'kernel':['linear'], 'degree':[3,4,5]},
    {'C':[1,10,100], 'kernel':['rbf', 'linear'], 'gamma':[0.001, 0.0001]},
    {'C':[1,10,100,1000], 'kernel':['sigmoid'], 'gamma':[0.01, 0.001, 0.0001], 'degree':[3, 4]},
    {'C':[0.1, 1], 'gamma':[1, 10]}
]

# GridSearch CV default : StratifiedKFold 
model = HalvingGridSearchCV(SVC(), parameters, cv=5, verbose=1, refit=True, n_jobs=-1, factor=2.4)
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print('time : ', round(end_time-start_time, 2),'s')
# time :  10.15 s
# time :  4.3 s (Halving)

# print(x.shape, x_train.shape)       # (1797, 64) (1437, 64)

y_predict = model.predict(x_test)
print('acc : ', accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('y_pred_best acc', accuracy_score(y_test, y_pred_best))

print('time : ', round(end_time-start_time, 2),'s')