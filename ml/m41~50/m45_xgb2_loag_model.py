import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
# import pickle
import joblib
 
# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8, stratify=y)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
path = './_save/pickle_test/'
# model = pickle.load(open(path + 'm43_pickle1_save.dat', 'rb'))
# model = joblib.load(path + 'm43_pickle1_save.dat')
model = XGBClassifier()
model.load_model(path + 'm44_joblib1_save.dat')

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('result : ', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)