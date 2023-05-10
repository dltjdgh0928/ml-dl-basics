import pandas as pd
import numpy as np
import glob
from preprocess_3 import bring

path = './_data/finedust/'
path_save = './_save/finedust/'
pm_best_csv = pd.read_csv(path_save + 'fd_Submit_time_0509_1353.csv')
# pm_best_csv['month'] = pm_best_csv['일시'].str[:2].astype('int8')
print(pm_best_csv)
print(pm_best_csv.shape)

import datetime

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

pm_best_csv = np.array(pm_best_csv).reshape(17, -1, 4)
print(pm_best_csv)
print(pm_best_csv.shape)

for j in range(17):
    for i in range(64):
        pm_best_csv[j, 0+72*i:60+72*i, 3] = pm_best_csv[j, 0+72*i:60+72*i, 3] - 0.004
        pm_best_csv[j, 60+72*i:72+72*i, 3] = pm_best_csv[j, 60+72*i:72+72*i, 3] - 0.008

pm = pm_best_csv[:, :, 3].reshape(-1,)
anwser = pd.read_csv(path + 'answer_sample.csv', index_col=0)
anwser['PM2.5'] = pm
anwser.to_csv(path_save + 'submit' + date + '.csv')




# test_pm_path = glob.glob(path + 'TEST_INPUT/*.csv')

# test_pm = bring(test_pm_path)
# # test_pm['month'] = test_pm['일시'].str[:2].astype('int8')

# print(test_pm)
# print(test_pm.shape)

# test_pm = np.array(test_pm).reshape(17, -1, 5)

# print(test_pm.shape)


# pm = pm_best_csv[:, :, 3]
# real = test_pm[:, :, 3]

# print(pm)
# print(pm.shape)

# sum_mean = []
# for j in range(17):
#     a=[]
#     b=[]
#     for i in range(64):
#         a.append(pm[j, 72*i+71])
#         b.append(real[j, 120*i+120])
#     c = np.mean(a)
#     d = np.mean(b)
#     print(j, '번째 지역 예측 마지막값들 평균 : ', c)
#     print(j, '번째 지역 첫 실제 값들 평균 : ', d)
#     print(j, '번째 지역 차이', c-d)
    
    
    
# # print(real)

# for j in range(17):
#     a=[]
#     for i in range(64):
#         a.append(real[j, 120*i+120])


