import numpy as np
from sklearn.covariance import EllipticEnvelope

a = np.array([[-10, 2, 3, 4, 5, 6, 700, 8, 9, 10, 11, 12, 50],
             [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]]).transpose()

outliers = EllipticEnvelope(contamination=0.1)

for i in range(a.shape[1]):
    outliers.fit(a[:, i].reshape(-1, 1))
    results = outliers.predict(a[:, i].reshape(-1, 1))
    print(results)