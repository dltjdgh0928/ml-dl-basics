import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
from sklearn.metrics import mean_absolute_error, r2_score

# / // \ \\ 같다

path = './_data/pm2.5/'

# _data
#     TRAIN
#     TRAIN_AWS
#     TEST_INPUT
#     TEST_AWS
#     META
#     answer_sample.csv

train_files = glob.glob(path + 'TRAIN/*.csv')

test_input_files = glob.glob(path + 'test_input/*.csv')

######################################## Train 폴더 ################################################
li = []
for filename in train_files:
    df = pd.read_csv(filename, index_col=None, header=0, encoding='utf-8-sig')
    li.append(df)
train_dataset = pd.concat(li, axis=0, ignore_index=True)

######################################## Test 폴더 ################################################

li = []
for filename in test_input_files:
    df = pd.read_csv(filename, index_col=None, header=0, encoding='utf-8-sig')
    li.append(df)
test_input_dataset = pd.concat(li, axis=0, ignore_index=True)


####################################### 측정소 라벨인코더 ##########################################
le = LabelEncoder()
train_dataset['locate'] = le.fit_transform(train_dataset['측정소'])
test_input_dataset['locate'] = le.transform(test_input_dataset['측정소'])
train_dataset = train_dataset.drop(['측정소'], axis=1)
test_input_dataset = test_input_dataset.drop(['측정소'], axis=1)


####################################### 일시 -> 월, 시간으로 분리!! ############################
train_dataset['month'] = train_dataset['일시'].str[:2].astype('int8')
train_dataset['hour'] = train_dataset['일시'].str[6:8].astype('int8')
train_dataset = train_dataset.drop(['일시'], axis=1)

test_input_dataset['month'] = test_input_dataset['일시'].str[:2].astype('int8')
test_input_dataset['hour'] = test_input_dataset['일시'].str[6:8].astype('int8')
test_input_dataset = test_input_dataset.drop(['일시'], axis=1)

####################################### 결측치 제거 pm2.5에 15542개 있다. #######################
# 전체 596085 -> 580546
# print(train_dataset.info())

train_dataset = train_dataset.dropna()
print(train_dataset.info())


##### 시즌 - 파생피처도 생각해봐 ##########
y = train_dataset['PM2.5']
x = train_dataset.drop(['PM2.5'], axis=1)
print(x, '\n', y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=5090, shuffle=True)

parameters = {
    'n_estimators' : 3,
    'learning_rate' : 0.07,
    'max_depth' : 3,
    'gamma' : 1,
    'min_child_weight' : 1,
    'subsample' : 1,
    'colsample_bytree' : 1,
    'colsample_bylevel' : 1,
    'colsample_bynode' : 1,
    'reg_alpha' : 0,
    'reg_lambda' : 1,
    'random_state' : 337,
}

# 2. 모델
model = XGBRegressor()

# 3. 컴파일, 훈련
model.set_params(**parameters,
                 eval_metric='mae',
                 early_stopping_rounds=200,
                )
start_time = time.time()
model.fit(
    x_train, y_train, verbose=1, eval_set=[(x_train, y_train), (x_test, y_test)]
)
end_time = time.time()
print('time : ', round(end_time - start_time, 2), '초')

y_predict = model.predict(x_test)
results = model.score(x_test, y_test)
print('model.score : ', results)

r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

mae = mean_absolute_error(y_test, y_predict)
print('mae : ', mae)


x_submit = test_input_dataset[test_input_dataset.isna().any(axis=1)]
# print(x_submit)
# print(x_submit.info())

x_submit = x_submit.drop(['PM2.5'], axis=1)
print(x_submit)
y_submit = model.predict(x_submit)
answer_sample_csv = pd.read_csv(path + 'answer_sample.csv', index_col=None, header=0, encoding='utf-8-sig')
# print(answer_sample_csv)
# print(answer_sample_csv.info())
answer_sample_csv['PM2.5'] = y_submit
print(answer_sample_csv)

answer_sample_csv.to_csv(path + 'm50_factory2_submit.csv', index=None)



