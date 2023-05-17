import numpy as np
import pandas as pd

a = pd.read_csv('./_save/finedust/아둔토리다스913450516_2000.csv')
b = pd.read_csv('./_save/finedust/아둔토리다스913450516_2113.csv')

print(a)
print(b)

a = a.values
b = b.values
print(a)
print(b)

file1 = a[: , 3]
file1 = np.float32(file1)
print(file1)

file2 = b[:, 3]
file2 = np.float32(file2)
print(file2)

차 = file1 - file2
print(np.unique(차))