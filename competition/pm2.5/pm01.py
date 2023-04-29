import os
import numpy as np
import pandas as pd
from haversine import haversine
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
<<<<<<< HEAD
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
=======
>>>>>>> da223433e5a4b3ec783d888deb86038e322d9fc0
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

imputer = IterativeImputer(XGBRegressor())
le = OrdinalEncoder()

<<<<<<< HEAD
from preprocess import load_aws_and_pm
awsmap,pmmap=load_aws_and_pm()

from preprocess import load_distance_of_pm_to_aws
distance_of_pm_to_aws=load_distance_of_pm_to_aws(awsmap,pmmap)

from preprocess import min_dist_from_pm
result,min_i=min_dist_from_pm(distance_of_pm_to_aws,pmmap)
distance_of_pm_to_aws=distance_of_pm_to_aws.values
result=result.values
print(result)


########################################################## 2부 
=======
path='./_data/pm2.5/'
path_list=os.listdir(path)
print(f'datafolder_list:{path_list}')

meta='/'.join([path,path_list[1]])
meta_list=os.listdir(meta)

print(f'META_list:{meta_list}')
awsmap=pd.read_csv('/'.join([meta,meta_list[0]]))
awsmap=awsmap.drop(awsmap.columns[-1],axis=1)
pmmap=pd.read_csv('/'.join([meta,meta_list[1]]))
pmmap=pmmap.drop(pmmap.columns[-1],axis=1)

a = []
for i in range(pmmap.shape[0]):
    for j in range(awsmap.shape[0]):
        a.append(haversine((np.array(pmmap)[i, 1], np.array(pmmap)[i, 2]), (np.array(awsmap)[j, 1], np.array(awsmap)[j, 2])))

b = np.array(a).reshape(17, 30)

min_i=[]
min_v=[]

near = 3
for i in range(pmmap.shape[0]):
    min_i.append(np.argsort(b[i,:])[:near])
    min_v.append(b[i, min_i[i]])

min_i = np.array(min_i)
min_v = np.array(min_v)
for i in range(pmmap.shape[0]):
    for j in range(pmmap.shape[1]):
        min_v[i, j]=min_v[i, j]**2
        
sum_min_v = np.sum(min_v, axis=1)

recip=[]
for i in range(pmmap.shape[0]):
    recip.append(sum_min_v[i]/min_v[i, :])
recip = np.array(recip)
recip_sum = np.sum(recip, axis=1)
coef = 1/recip_sum

result = []
for i in range(pmmap.shape[0]):
    result.append(recip[i, :]*coef[i])
result = np.array(result)






>>>>>>> da223433e5a4b3ec783d888deb86038e322d9fc0
train_pm_path = './_data/pm2.5/TRAIN/'
train_aws_path = './_data/pm2.5/TRAIN_AWS/'
test_pm_path = './_data/pm2.5/TEST_INPUT/'
test_aws_path = './_data/pm2.5/TEST_AWS/'

<<<<<<< HEAD
def bring(path:str)->np.ndarray:
=======
def bring(path):
>>>>>>> da223433e5a4b3ec783d888deb86038e322d9fc0
    file_list = os.listdir(path)
    data_list = []
    for file_name in file_list:
        if file_name.endswith(".csv"):
            file_path = os.path.join(path, file_name)
            data = pd.read_csv(file_path).values
            data_list.append(data)
    data_array = np.array(data_list)
    return data_array

train_pm = bring(train_pm_path)
train_aws = bring(train_aws_path)
test_pm = bring(test_pm_path)
test_aws = bring(test_aws_path)

<<<<<<< HEAD
print(train_pm.shape)

for i in range(train_pm.shape[0]):
    train_pm[i, :, 3] = imputer.fit_transform(train_pm[i, :, 3].reshape(-1, 1)).reshape(-1,)
# print(pd.DataFrame(train_pm.reshape(-1,4)).isna().sum())
=======
print(train_pm[0, :, 3])
print(train_pm[0, :, 3].shape)

print(imputer.fit_transform(train_pm[i, :, 3].reshape(-1, 1)))
print(imputer.fit_transform(train_pm[i, :, 3].reshape(-1, 1)).shape)


for i in range(train_pm.shape[0]):
    train_pm[i, :, 3] = imputer.fit_transform(train_pm[i, :, 3].reshape(-1, 1)).reshape(-1,)
print(pd.DataFrame(train_pm.reshape(-1,4)).isna().sum())
>>>>>>> da223433e5a4b3ec783d888deb86038e322d9fc0

for j in range(train_aws.shape[2]-3):
    for i in range(train_aws.shape[0]):
        train_aws[i, :, j+3] = imputer.fit_transform(train_aws[i, :, j+3].reshape(-1, 1)).reshape(-1,)
<<<<<<< HEAD
# print(pd.DataFrame(train_aws.reshape(-1,8)).isna().sum())

train_pm = train_pm.reshape(-1, 4)[:, 2:]
test_pm =  test_pm.reshape(-1, 4)[:, 2:]
train_aws =  train_aws.reshape(-1, 8)[:, 2:]
test_aws =  test_aws.reshape(30, -1, 8)[:, :, 2:]

print(train_pm.shape)
=======

print(pd.DataFrame(train_aws.reshape(-1,8)).isna().sum())


print(train_pm.shape)
print(train_aws.shape)
print(test_pm.shape)
print(test_aws.shape)
print(pd.DataFrame(test_aws[0]).isna().sum())

train_pm[:, :, 3] = imputer.fit_transform(train_pm[:, :, 3])
print(train_pm[:, :, 3])
print(pd.DataFrame(train_pm[:, :, 3]).isna().sum())
train_pm = train_pm.reshape(-1, 4)[:, 2:]
test_pm = test_pm.reshape(-1, 4)[:, 2:]
train_aws = train_aws.reshape(-1, 8)[:, 2:]
test_aws = test_aws.reshape(-1, 8)[:, 2:]
>>>>>>> da223433e5a4b3ec783d888deb86038e322d9fc0

def label(x):
    label_dict = {}
    labels = []
    for i in x:
        if i not in label_dict:
            label_dict[i] = len(label_dict)
        labels.append(label_dict[i])
    return labels

<<<<<<< HEAD

train_pm[:, 0] = label(train_pm[:, 0])
train_aws[:, 0] = label(train_aws[:, 0])
test_pm[:, 0] = label(test_pm[:, 0])

train_pm = train_pm.reshape(17, -1, 2)
test_pm = test_pm.reshape(17, -1, 2)
train_aws = train_aws.reshape(30, -1, 6)

for i in range(test_aws.shape[0]):
    test_aws[i, :, 0] = pd.DataFrame(test_aws[i, :, 0]).ffill().values.reshape(-1,)
test_aws = test_aws.reshape(-1, 6)
test_aws[:, 0] = label(test_aws[:, 0])
test_aws = test_aws.reshape(30, -1, 6)

=======
train_pm[:, 0] = label(train_pm[:, 0])
test_pm[:, 0] = label(test_pm[:, 0])
train_aws[:, 0] = label(train_aws[:, 0])
test_aws[:, 0] = label(test_aws[:, 0])

# pd.DataFrame(train_pm).to_csv('./_save/pm2.5/' + 'check4.csv')
>>>>>>> da223433e5a4b3ec783d888deb86038e322d9fc0
print(train_pm.shape)
print(train_aws.shape)
print(test_pm.shape)
print(test_aws.shape)
<<<<<<< HEAD





train_aws = train_aws.astype(float)

train_pm_aws=[]
for i in range(train_pm.shape[0]):
    train_pm_aws.append(train_aws[min_i[i, 0], :, 1:]*result[0, 0] + train_aws[min_i[i, 1], :, 1:]*result[0, 1] + train_aws[min_i[i, 2], :, 1:]*result[0, 2])

print(train_pm_aws)
train_pm_aws = np.array(train_pm_aws)









train_data = np.concatenate([train_pm, train_pm_aws], axis=2)
print(train_data)
print(train_data.shape)

def split_x(dt, ts):
    a = []
    for j in range(dt.shape[0]):
        b = []
        for i in range(dt.shape[1]-ts):
            c = dt[j, i:i+ts, :]
            b.append(c)
        a.append(b)
    return np.array(a)

timesteps = 10

x = split_x(train_data, timesteps).reshape(-1, timesteps, train_data.shape[2])
print(x)
print(x.shape)

y = []
for i in range(train_data.shape[0]):
    y.append(train_data[i, timesteps:, 1].reshape(train_data.shape[1]-timesteps,))

y = np.array(y).reshape(-1,)
print(y)
print(y.shape)

x_train, y_train, x_test, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)
# scaler = MinMaxScaler()
# x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)

model = Sequential()
model.add(LSTM(32, input_shape=(timesteps, train_data.shape[2])))
model.add(Dense(16))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, batch_size=128, epochs=100)

model.evaluate(x_test, y_test)
=======
print(pd.DataFrame(train_pm).isna().sum())
print(pd.DataFrame(train_aws).isna().sum())
print(pd.DataFrame(test_pm).isna().sum())
print(pd.DataFrame(test_aws).isna().sum())


train_pm = imputer.fit_transform(train_pm)
train_aws = imputer.fit_transform(train_aws)

print(pd.DataFrame(train_pm).isna().sum())
print(pd.DataFrame(train_aws).isna().sum())
>>>>>>> da223433e5a4b3ec783d888deb86038e322d9fc0
