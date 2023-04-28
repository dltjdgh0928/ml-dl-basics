# graph plot
# 1. value_counts 쓰지마
# 2. np.unique return_counts 쓰지마
####### 3. groupby 써, count 써

# plt.bar 로 그린다 (quality 컬럼)

# 힌트
# 데이터갯수(y축) = 데이터갯수.주저리 주저리

# outlier

import pandas as pd
import numpy as np
import optuna
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn. ensemble import RandomForestClassifier
import datetime
import warnings
warnings.filterwarnings('ignore')
def outliers(a):
    b = []
    for i in range(a.shape[1]):
        q1, q3 = np.percentile(a[:, i], [25, 75], axis=0)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - (iqr * 1.5), q3 + (iqr * 1.5)
        b.append(np.where((a[:, i]>upper_bound)|(a[:, i]<lower_bound))[0])
    return b
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

import matplotlib.pyplot as plt
group = np.array(wine.groupby('quality').count())
print(group)
print(group[:,0])
plt.bar(np.arange(7)+3, group[:,0])
plt.title('quality')
plt.show()


y = wine['quality'] - 3

out_wine = outliers(x)
for i in range(x.shape[1]):
    x[out_wine[i], i] = None
    x = imputer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)

model = RandomForestClassifier()

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('result : ', results)

