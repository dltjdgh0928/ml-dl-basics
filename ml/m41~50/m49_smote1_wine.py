import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

datasets = load_wine()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)         # (178, 13) (178,)
print(np.unique(y, return_counts=True))
print(pd.Series(y).value_counts().sort_index())
# 0    59
# 1    71
# 2    48
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
x = x[:-25]
y = y[:-25]
print(x.shape, y.shape)         # (153, 13) (153,)
print(y)
print(pd.Series(y).value_counts().sort_index())
# 0    59
# 1    71
# 2    23

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=3377, stratify=y)
print(pd.Series(y_train).value_counts().sort_index())

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
print('score : ', score)
print('acc : ', accuracy_score(y_test, y_pred))
print('f1(macro) : ', f1_score(y_test, y_pred, average='macro'))
print('f1(micro) : ', f1_score(y_test, y_pred, average='micro'))

smote = SMOTE(random_state=3377, k_neighbors=3)
x_train, y_train = smote.fit_resample(x_train, y_train)
print(x_train.shape, y_train.shape)
print(pd.Series(y_train).value_counts().sort_index())
# 0    53
# 1    53
# 2    53
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
print('score : ', score)
print('acc : ', accuracy_score(y_test, y_pred))
print('f1(macro) : ', f1_score(y_test, y_pred, average='macro'))
print('f1(micro) : ', f1_score(y_test, y_pred, average='micro'))
