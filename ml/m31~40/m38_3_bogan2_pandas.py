import numpy as np
import pandas as pd

# data = pd.DataFrame([[2, np.nan, 6, 8, 10],
#                     [2, 4, np.nan, 8, np.nan],
#                     [2, 4, 6, 8, 10],
#                     [np.nan, 4, np.nan, 8, np.nan]]).transpose()

data = pd.DataFrame([[2, 2, 2, np.nan], [np.nan, 4, 4, 4], [6, np.nan, 6, np.nan], [8, 8, 8, 8], [10, np.nan, 10, np.nan]])
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

data2 = data.copy()
# print(data.shape)

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# (5, 4)

# 0. 결측치 확인
# print(data.isnull().sum())
# print(data.info())

# 1. 결측지 삭제
# print(data['x1'].dropna())
# print(data.dropna())                # delete row ( default )
# print(data.dropna(axis=1))          # delete col

# 2-1. 특정값 - 평균
# means = data.mean()
# print(means)
# print(data.fillna(means))

# 2-2. 특정값 - 중앙값
# median = data.median()
# print(median)
# print(data.fillna(median))

# 2-3. 특정값 - ffill, bfill
# print(data.ffill())
# print(data.bfill())

# 2-4. 특정값 - 임의의 값으로 채우기
# data6 = data.fillna(value=777777)
# print(data6)

# 특정 칼럼만 !!! #
# 1. x1 컬럼에 평균값
data['x1'] = data['x1'].fillna(data['x1'].mean())

# 2. x2 컬럼에 중앙값
data['x2'] = data['x2'].fillna(data['x2'].median())

# 3. x4 컬럼에 ffill + 777777
data['x4'] = data['x4'].ffill()
data['x4'] = data['x4'].fillna(777777)
print(data)

data2['x1'] = data2['x1'].interpolate()
data2['x2'] = data2['x2'].interpolate()
data2['x4'] = data2['x4'].interpolate()
data2['x4'] = data2['x4'].ffill()
data2['x4'] = data2['x4'].bfill()
print(data2)