import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score


# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8, stratify=y)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)
parameters = {'n_estimators' : 1000,
              'learning_rate' : 0.3,
              'max_depth' : 3,
              'gamma' : 1,
              'min_child_weight' : 1,
              'subsample' : 1,
              'colsample_bytree' : 1,
              'colsample_bylevel' : 1,
              'colsample_bynode' : 1,
              'reg_alpha' : 0,
              'reg_lambda' : 1,
              'random_state' : 337,
              }

# 2. 모델
model = XGBClassifier(**parameters)

# 3. 훈련
hist = model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          early_stopping_rounds=10,
          verbose=1,
        #   eval_metric='logloss',          # 이진분류
          eval_metric='error',            # 이진분류
        #   eval_metric='auc',              # 이진분류
        #   eval_metric='merror',           # 다중분류
        #   eval_metric='mlogloss',           # 다중분류
        #   eval_metric='rmse', 'mae', 'rmsle', ...    # 회귀
          )

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('result : ', results)
hist = model.evals_result()

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)

import pickle
path = './_save/pickle_test/'
pickle.dump(model, open(path + 'm43_pickle1_save.dat', 'wb'))