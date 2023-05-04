import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler

x, y = make_blobs(random_state=337,
                  n_samples=50,         # 클러스터 개수 / y의 라벨
                  centers=2,            # 클러스터 표준편차
                  cluster_std=1
                  )
# 가우시안 정규분포 샘플 생성


print(x)
print(y)
print(x.shape, y.shape)

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].scatter(x[:, 0], x[:, 1:], c=y, edgecolors='black')
ax[0, 0].set_title('original')

scaler = QuantileTransformer(n_quantiles=50)
x_trans = scaler.fit_transform(x)
ax[0, 1].scatter(x_trans[:, 0], x_trans[:, 1:], c=y, edgecolors='black')
ax[0, 1].set_title(type(scaler).__name__)

scaler = PowerTransformer()
x_trans = scaler.fit_transform(x)
ax[1, 0].scatter(x_trans[:, 0], x_trans[:, 1:], c=y, edgecolors='black')
ax[1, 0].set_title(type(scaler).__name__)

scaler = StandardScaler()
x_trans = scaler.fit_transform(x)
ax[1, 1].scatter(x_trans[:, 0], x_trans[:, 1:], c=y, edgecolors='black')
ax[1, 1].set_title(type(scaler).__name__)

plt.show()
