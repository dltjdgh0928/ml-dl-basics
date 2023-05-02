import numpy as np
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import mean_absolute_error


x, y = load_linnerud(return_X_y=True)
print(x)
print(y)
print(x.shape)
print(y.shape)

model = Ridge()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, 'score : ', round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))

model = XGBRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__,'score : ', round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))

model = MultiOutputRegressor(LGBMRegressor())
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__,'score : ', round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))

model = CatBoostRegressor(verbose=0, loss_function='MultiRMSE')
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__,'score : ', round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))
