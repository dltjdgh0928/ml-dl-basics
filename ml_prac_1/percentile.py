import numpy as np

a = np.array([-10, 5, 7, 9, 10, 12, 15, 17, 18, 24, 34, 44, 50])
qa1, qa2, qa3 = np.percentile(a, [25, 50, 75])
print(qa1, qa2, qa3)

b = np.array([24, -10, 7, 9, 44, 55, 10, 17, 12, 5, 15, 18, 34, 50])
qb1, qb2, qb3 = np.percentile(b, [25, 50, 75])
print(qb1, qb2, qb3)