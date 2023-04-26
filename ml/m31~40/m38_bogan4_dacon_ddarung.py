import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

imputer = IterativeImputer(estimator=XGBRegressor())
path = './_data/ddarung/'
ddarung = pd.read_csv(path + 'train.csv', index_col=0)
ddarung = imputer.fit_transform(ddarung)
print(pd.DataFrame(ddarung))
x = pd.DataFrame(ddarung).drop(9, axis=1)
y = pd.DataFrame(ddarung)[9]

model = XGBRegressor()
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)
model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print('result : ', result)