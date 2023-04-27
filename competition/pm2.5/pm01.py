import pandas as pd
import os
from haversine import haversine
import numpy as np
path='./_data/pm2.5/'
path_list=os.listdir(path)
print(f'datafolder_list:{path_list}')

meta='/'.join([path,path_list[1]])
meta_list=os.listdir(meta)
test_aws='/'.join([path,path_list[2]])
test_aws_list=os.listdir(test_aws)
test_input='/'.join([path,path_list[3]])
test_input_list=os.listdir(test_input)
train='/'.join([path,path_list[4]])
train_list=os.listdir(train)
train_aws='/'.join([path,path_list[5]])
train_aws_list=os.listdir(train_aws)

print(f'META_list:{meta_list}')
awsmap=pd.read_csv('/'.join([meta,meta_list[0]]))
awsmap=awsmap.drop(awsmap.columns[-1],axis=1)
pmmap=pd.read_csv('/'.join([meta,meta_list[1]]))
pmmap=pmmap.drop(pmmap.columns[-1],axis=1)
print(awsmap)
print(pmmap)
print(pmmap.shape)
# print(test_aws_list)
# print(test_input_list)
# print(train_list)
# print(train_aws_list)

a = []
for i in range(pmmap.shape[0]):
    for j in range(awsmap.shape[0]):
        a.append(haversine((np.array(pmmap)[i, 1], np.array(pmmap)[i, 2]), (np.array(awsmap)[j, 1], np.array(awsmap)[j, 2])))

print(a)
print(np.array(a).reshape(17, 30))