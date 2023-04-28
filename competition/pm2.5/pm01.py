import os
import numpy as np
import pandas as pd
from haversine import haversine
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

imputer = IterativeImputer(XGBRegressor())
le = OrdinalEncoder()

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






train_pm_path = './_data/pm2.5/TRAIN/'
train_aws_path = './_data/pm2.5/TRAIN_AWS/'
test_pm_path = './_data/pm2.5/TEST_INPUT/'
test_aws_path = './_data/pm2.5/TEST_AWS/'

def bring(path):
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

print(train_pm[0, :, 3])
print(train_pm[0, :, 3].shape)

print(imputer.fit_transform(train_pm[i, :, 3].reshape(-1, 1)))
print(imputer.fit_transform(train_pm[i, :, 3].reshape(-1, 1)).shape)


for i in range(train_pm.shape[0]):
    train_pm[i, :, 3] = imputer.fit_transform(train_pm[i, :, 3].reshape(-1, 1)).reshape(-1,)
print(pd.DataFrame(train_pm.reshape(-1,4)).isna().sum())

for j in range(train_aws.shape[2]-3):
    for i in range(train_aws.shape[0]):
        train_aws[i, :, j+3] = imputer.fit_transform(train_aws[i, :, j+3].reshape(-1, 1)).reshape(-1,)

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

def label(x):
    label_dict = {}
    labels = []
    for i in x:
        if i not in label_dict:
            label_dict[i] = len(label_dict)
        labels.append(label_dict[i])
    return labels

train_pm[:, 0] = label(train_pm[:, 0])
test_pm[:, 0] = label(test_pm[:, 0])
train_aws[:, 0] = label(train_aws[:, 0])
test_aws[:, 0] = label(test_aws[:, 0])

# pd.DataFrame(train_pm).to_csv('./_save/pm2.5/' + 'check4.csv')
print(train_pm.shape)
print(train_aws.shape)
print(test_pm.shape)
print(test_aws.shape)
print(pd.DataFrame(train_pm).isna().sum())
print(pd.DataFrame(train_aws).isna().sum())
print(pd.DataFrame(test_pm).isna().sum())
print(pd.DataFrame(test_aws).isna().sum())


train_pm = imputer.fit_transform(train_pm)
train_aws = imputer.fit_transform(train_aws)

print(pd.DataFrame(train_pm).isna().sum())
print(pd.DataFrame(train_aws).isna().sum())