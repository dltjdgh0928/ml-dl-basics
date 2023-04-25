# Linear Discriminant Analysis ( 지도 학습임, pca는 비지도 )

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype

data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype]
lda = LinearDiscriminantAnalysis()

for i in data_list:
    x, y = i(return_X_y=True)
    x_lda = lda.fit_transform(x, y)    
    lda_EVR = lda.explained_variance_ratio_
    cumsum = np.cumsum(lda_EVR)
    print(i.__name__, f'cumsum : {cumsum}, \nx.shape : {x.shape}, \nx_lda.shape : {x_lda.shape}')
    