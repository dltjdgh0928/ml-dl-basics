# Linear Discriminant Analysis ( 지도 학습임, pca는 비지도 )

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits
from tensorflow.keras.datasets import cifar100


# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape)
print(y_train.shape)
x_train = x_train.reshape(50000, 32*32*3)


pca = PCA(n_components=85)
x_train = pca.fit_transform(x_train)

lda = LinearDiscriminantAnalysis()
x_lda = lda.fit_transform(x_train, y_train)

# default = min(n_classes - 1, n_features)

print(x_lda.shape)