from sklearn.datasets import fetch_california_housing, load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
datasets = load_diabetes()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target
print(df)

# df.boxplot()
# df.plot.box()
# plt.show()

# df.info()
# print(df.describe())

df['target'].hist(bins=50)
plt.show()

y = df['target']
x = df.drop(['target'], axis=1)

x['age'] = np.log1p(x['age'])
y = np.log1p(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=337)

model = RandomForestRegressor(random_state=337)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('r2 : ', r2_score(np.expm1(y_test), np.expm1(model.predict(x_test))))
print('score : ', score)