import os
import numpy as np
import pandas as pd
from haversine import haversine
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

imputer = IterativeImputer(XGBRegressor())
le = LabelEncoder()

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






import os
import pandas as pd
import numpy as np

# csv 파일이 들어있는 폴더 경로
folder_path = "/path/to/folder/"

# csv 파일 목록 불러오기
file_list = os.listdir(folder_path)

# 각 파일을 읽어와서 2D NumPy 배열로 변환한 후 리스트에 추가
data_list = []
for file_name in file_list:
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(file_path).values
        data_list.append(data)

# 3D NumPy 배열로 합치기
data_array = np.array(data_list)

print(data_array.shape)


# path_train_pm = './_data/pm2.5/TRAIN/'
# path_train_aws = './_data/pm2.5/TRAIN_AWS/'

# train_아름동_pm = pd.DataFrame(imputer.fit_transform(pd.read_csv(path_train_pm + '아름동.csv', index_col=0).drop(['일시', '측정소'], axis=1)))
# print(train_아름동_pm.shape)

# train_세종고운_aws = pd.DataFrame(imputer.fit_transform(pd.read_csv(path_train_aws + '세종고운.csv', index_col=0).drop(['일시', '지점'], axis=1)))
# train_세종금남_aws = pd.DataFrame(imputer.fit_transform(pd.read_csv(path_train_aws + '세종금남.csv', index_col=0).drop(['일시', '지점'], axis=1)))
# train_세종연서_aws = pd.DataFrame(imputer.fit_transform(pd.read_csv(path_train_aws + '세종연서.csv', index_col=0).drop(['일시', '지점'], axis=1)))
# print(train_세종연서_aws.shape)
# print(train_세종연서_aws.isna().sum())




# def bring(path):
#     data_list = []
#     for filename in os.listdir(path):
#         if filename.endswith('.csv'):
#             with open(os.path.join(path, filename), 'r') as f:
#                 data=f.read()
#                 data_list.append(data)
#                 rows = data.split('\n')  # 개행문자('\n')를 기준으로 문자열을 분리하여 리스트로 만듦
#                 # print(rows)
#                 rows = [row.split(',') for row in rows]  # 각 행을 쉼표(,)로 구분하여 분리
#                 # print(rows)
#                 # rows.pop()  # 마지막 행은 빈 문자열이므로 제거
#                 arr = []
#                 for i in rows:
#                     arr.append(rows)
                
#                 # for row in rows:
#                 #     values = [float(value) for value in row[3:]]  # 문자열을 실수형으로 변환하여 추출
#                 #     arr.append(values)  # 값을 리스트에 추가
#                 # arr = np.array(arr)  # 리스트를 배열로 변환
#                 # print(arr.shape)  # 출력: (n, 8)
#     print(len(data_list))
#     return data_list

# train_aws = './_data/pm2.5/TRAIN_AWS/'
# train_aws = bring(train_aws)
# print(train_aws)

