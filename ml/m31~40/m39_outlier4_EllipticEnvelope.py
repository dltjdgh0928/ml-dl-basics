import numpy as np
a = np.array([-10, 2, 3, 4, 5, 6, 700,
                8, 9, 10, 11, 12, 50]).reshape(-1,1)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=0.0834)
# 1/12 = 0.0833333... 이니까 0.0834부터 2개 오염

outliers.fit(a)
results = outliers.predict(a)
print(results)