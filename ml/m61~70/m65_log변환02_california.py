from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target
print(df)

# df.boxplot()
# df.plot.box()
# plt.show()

# df.info()
# print(df.describe())

# df['Population'].boxplot()
# df['Population'].plot.box()
# plt.show()

# df['Population'].hist(bins=50)
# plt.show()

# df['target'].hist(bins=50)
# plt.show()

y = df['target']
x = df.drop(['target'], axis=1)

############### x population 로그변환 ###############
x['Population'] = np.log1p(x['Population'])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=337)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

model = RandomForestRegressor(random_state=337)

model.fit(x_train, y_train_log)

score = model.score(x_test, y_test_log)
print('score : ', score)

print('로그 -> 지수 r2 : ', r2_score(y_test, np.expm1(model.predict(x_test))))