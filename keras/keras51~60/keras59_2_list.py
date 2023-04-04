import numpy as np
import pandas as pd

a = [[1,2,3], [4,5,6]]
print(a)
print(type(a))
b = np.array(a)
print(b)

c = [[1,2,3], [4,5]]
print(c)

d = np.array(c)
print(d)

# 1. 리스트는 크기가 달라도 상관이 없다

#####################################################################
e = [[1,2,3], ['바보', '맹구', 5, 6]]
print(e)

f = np.array(e)
print(f)