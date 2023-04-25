import numpy as np
import pandas as pd
import sklearn as sk
print(sk.__version__)       # 1.0.2

# data = pd.DataFrame([[2, np.nan, 6, 8, 10],
#                     [2, 4, np.nan, 8, np.nan],
#                     [2, 4, 6, 8, 10],
#                     [np.nan, 4, np.nan, 8, np.nan]]).transpose()

data = pd.DataFrame([[2, 2, 2, np.nan], [np.nan, 4, 4, 4], [6, np.nan, 6, np.nan], [8, 8, 8, 8], [10, np.nan, 10, np.nan]])
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer      # 결측치에 대한 책임
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor
# imputer = IterativeImputer()

# imputer = SimpleImputer()                                     # 디폴트 평균
# imputer = SimpleImputer(strategy='mean')                      # 평균값
# imputer = SimpleImputer(strategy='median')                    # 중앙값
# imputer = SimpleImputer(strategy='most_frequent')             # 최빈값(갯수 똑같으면 가장 작은값)
# imputer = SimpleImputer(strategy='constant', fill_value=77)   # 상수
# imputer = KNNImputer()

# imputer = IterativeImputer(estimator=DecisionTreeRegressor())
imputer = IterativeImputer(estimator=XGBRegressor())

data2 = imputer.fit_transform(data)
print(data2)
