# Linear Discriminant Analysis ( 지도 학습임, pca는 비지도 )

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_diabetes, fetch_california_housing
from tensorflow.keras.datasets import cifar100


# 1. 데이터
x, y = load_diabetes(return_X_y=True)
print(x.shape)
print(y)
print(np.unique(y, return_counts=True))

# x, y = fetch_california_housing(return_X_y=True)
# print(x.shape)
# y = np.round(y)
# print(np.unique(y))

lda = LinearDiscriminantAnalysis()
x_lda = lda.fit_transform(x, y)
print(x_lda.shape)

# 결론 : 회귀는 쓰면 안된다.
