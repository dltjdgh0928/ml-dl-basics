from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data()
del _

# x = np.concatenate([x_train, x_test], axis=0)
x = np.append(x_train, x_test, axis=0)
print(x.shape)

# pca를 통해 0.95 이상인 n_components 몇개
# 0.99
# 0.999
# 1.0
x = x.reshape(x.shape[0], -1)
pca = PCA(n_components=784)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_

cumsum = np.cumsum(pca_EVR)

print(np.argmax(cumsum >= 0.95)+1)        # 154
print(np.argmax(cumsum >= 0.99)+1)        # 331
print(np.argmax(cumsum >= 0.999)+1)        # 486
print(np.argmax(cumsum >= 1.0)+1)        # 713