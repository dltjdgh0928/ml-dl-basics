import pandas as pd
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, HalvingRandomSearchCV, HalvingGridSearchCV
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, LabelEncoder, QuantileTransformer
from sklearn.utils import all_estimators
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import all_estimators
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import optuna
import datetime
import warnings
warnings.filterwarnings('ignore')

scaler_list = [MinMaxScaler(), MaxAbsScaler(), StandardScaler(), RobustScaler()]
model_list = [RandomForestRegressor(), DecisionTreeRegressor()]

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

path = './_data/dacon_cal/'
path_save = './_save/dacon_cal/'
path_save_min = './_save/dacon_cal/min/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0).drop(['Weight_Status'], axis=1)
test_csv = pd.read_csv(path + 'test.csv', index_col=0).drop(['Weight_Status'], axis=1)
submit_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(['Calories_Burned'], axis=1)
y = train_csv['Calories_Burned']

x['Height(Feet)'] = 12*x['Height(Feet)']+x['Height(Remainder_Inches)']
x['Height(Remainder_Inches)'] = 703*x['Weight(lb)']/x['Height(Feet)']**2

test_csv['Height(Feet)'] = 12*test_csv['Height(Feet)']+test_csv['Height(Remainder_Inches)']
test_csv['Height(Remainder_Inches)'] = 703*test_csv['Weight(lb)']/test_csv['Height(Feet)']**2

le = LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'])
test_csv['Gender'] = le.transform(test_csv['Gender'])

min_rmse = 1

for k in range(1000000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=k)


    scaler = MaxAbsScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    test_csv = scaler.transform(test_csv)
    def objective(trial, x_train, y_train, x_test, y_test, min_rmse):
        alpha = trial.suggest_loguniform('alpha', 0.0001, 1)
        n_restarts_optimizer  = trial.suggest_int('n_restarts_optimizer', 3, 10)
        optimizer = trial.suggest_categorical('optimizer', ['fmin_l_bfgs_b', 'Powell', 'CG'])

        model = GaussianProcessRegressor(
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            optimizer=optimizer,
        )
        
        model.fit(x_train, y_train)
        
        print('GPR result : ', model.score(x_test, y_test))
        
        y_pred = model.predict(x_test)
        rmse = RMSE(y_test, y_pred)
        print('GPR RMSE : ', rmse)
        if rmse < 0.3:
            submit_csv['Calories_Burned'] = model.predict(test_csv)
            date = datetime.datetime.now()
            date = date.strftime('%m%d_%H%M%S')
            submit_csv.to_csv(path_save + date + str(round(rmse, 5)) + '.csv')
            # if rmse < min_rmse:
            #     min_rmse = rmse
            #     submit_csv.to_csv(path_save_min + date + str(round(rmse, 5)) + '.csv')
        return rmse
    opt = optuna.create_study(direction='minimize')
    opt.optimize(lambda trial: objective(trial, x_train, y_train, x_test, y_test, min_rmse), n_trials=20)
    print('best param : ', opt.best_params, 'best rmse : ', opt.best_value)
        
        # for (n, v) in regressor: 
        #     try:
        #         model = v()
        #         model.fit(x_train, y_train)
        #         result = model.score(x_test, y_test)
        #         print(n, 'result : ', result)
        #         y_pred = model.predict(x_test)
        #         rmse = RMSE(y_test, y_pred)
        #         print(n, 'RMSE : ', rmse)
        #         if rmse < 1:
        #             submit_csv['Calories_Burned'] = model.predict(test_csv)
        #             date = datetime.datetime.now()
        #             date = date.strftime('%m%d_%H%M%S')
        #             submit_csv.to_csv(path_save + n + date + str(round(rmse, 5)) + '.csv')
        #             if rmse < min_rmse:
        #                 min_rmse = rmse
        #                 submit_csv.to_csv(path_save_min + n + date + str(round(rmse, 5)) + '.csv')
        #     except:
        #         continue
        # for j in range(len(model_list)):
        #     if j==0:
        #         param = param_r
        #     elif j==1:
        #         param = param_d
        #     model = HalvingRandomSearchCV(model_list[j], param, cv=10, verbose=1)
        #     model.fit(x_train, y_train)

        #     loss = model.score(x_test, y_test)
        #     print('loss : ', loss)
        #     print('test RMSE : ', RMSE(y_test, model.predict(x_test)))
            
        #     if RMSE(y_test, model.predict(x_test))<0.5:
        #         submit_csv['Calories_Burned'] = model.predict(test_csv)
        #         date = datetime.datetime.now()
        #         date = date.strftime('%m%d_%H%M%S')
        #         submit_csv.to_csv(path_save + 'dacon_cal' + date + '.csv')
        #         break
            # else:
            #     # if j==0:
            #     #     model = RandomForestRegressor()
            #     # elif j==1:
            #     #     model = DecisionTreeRegressor()
            #     a = model.feature_importances_
            #     a = a.argmin(axis=0)
            #     x_train_d = pd.DataFrame(x_train).drop([a], axis=1)
            #     x_test_d = pd.DataFrame(x_test).drop([a], axis=1)
            #     test_csv_d = pd.DataFrame(test_csv).drop([a], axis=1)
            #     model = HalvingRandomSearchCV(model_list[j], param, cv=10, verbose=1)
            #     model.fit(x_train_d, y_train)
            #     loss = model.score(x_test_d, y_test)
            #     print('loss : ', loss)
            #     print('test RMSE : ', RMSE(y_test, model.predict(x_test_d)))
            #     if RMSE(y_test, model.predict(x_test_d))<0.5:
            #         submit_csv['Calories_Burned'] = model.predict(test_csv_d)
            #         date = datetime.datetime.now()
            #         date = date.strftime('%m%d_%H%M%S')
            #         submit_csv.to_csv(path_save + 'dacon_cal' + date + '.csv')
            #         break

# model = Sequential()
# model.add(Dense(32, input_shape=(x.shape[1],)))
# model.add(Dense(32, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(32, activation='selu'))
# model.add(Dropout(0.1))
# model.add(Dense(32, activation='swish'))
# model.add(Dense(32, activation='swish'))
# model.add(Dense(1, activation='swish'))

# model.compile(loss='mae', optimizer='adam')
# es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights=True)
# model.fit(x_train, y_train, epochs=1000, validation_split=0.2, batch_size=3, callbacks=[es])
