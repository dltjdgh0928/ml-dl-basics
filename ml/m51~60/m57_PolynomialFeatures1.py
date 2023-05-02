import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4, 2)

print(x)

pf = PolynomialFeatures(degree=2)
x_pf = pf.fit_transform(x)
print(x_pf)
print(x_pf.shape)

pf3 = PolynomialFeatures(degree=3)
x3 = pf3.fit_transform(x)
print(x3)
print(x3.shape)

pf4 = PolynomialFeatures(degree=4)
x4 = pf4.fit_transform(x)
print(x4)
print(x4.shape)

pf5 = PolynomialFeatures(degree=5)
x5 = pf5.fit_transform(x)
print(x5)
print(x5.shape)

pf6 = PolynomialFeatures(degree=6)
x6 = pf6.fit_transform(x)
print(x6)
print(x6.shape)


# col3, degree2
y = np.arange(12).reshape(4, 3)
print(y)
print(y.shape)

y2 = pf.fit_transform(y)
print(y2)
print(y2.shape)

y3 = pf3.fit_transform(y)
print(y3)
print(y3.shape)

y4 = pf4.fit_transform(y)
print(y4)
print(y4.shape)

y5 = pf5.fit_transform(y)
print(y5)
print(y5.shape)

y6 = pf6.fit_transform(y)
print(y6)
print(y6.shape)
