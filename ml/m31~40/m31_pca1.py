import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
datasets = load_diabetes()
x=datasets['data']
y=datasets.target
# print(x.shape, y.shape)       # (442, 10) (442,)

pca = PCA(n_components=9)
x = pca.fit_transform(x)
print(x.shape)                  # (442, 5)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)

# 2. 모델
model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('결과 : ', results)



