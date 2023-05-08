# import pandas as pd
# import numpy as np
# import glob

# path = './_data/finedust/'
# from preprocess_3 import bring

# train_pm_path = glob.glob(path + 'TRAIN/*.csv')
# train_pm = bring(train_pm_path)
# print('train pm mean : ', np.mean(train_pm['PM2.5'].dropna().values))
# test_pm_path = glob.glob(path + 'TEST_INPUT/*.csv')
# test_pm = bring(test_pm_path)
# print('test pm mean : ', np.mean(test_pm['PM2.5'].dropna().values))

# train_pm_num = np.unique(train_pm['PM2.5'].values, return_counts=True)

# best = pd.read_csv('./_save/finedust/_Submit_timebest.csv')
# print('best train mean : ', np.mean(best))
# best_num = np.unique(np.round(best['PM2.5'].values/0.004)*0.004, return_counts=True)

# import matplotlib.pyplot as plt
# plt.plot(train_pm_num[0], train_pm_num[1], label='train_file')
# plt.plot(best_num[0], best_num[1], label='submit_file_8.05')
# plt.figure('season')
# plt.plot(range(len(best['PM2.5'].values)),train_pm['PM2.5'].values[:len(best['PM2.5'].values)],label='train_file')
# plt.plot(range(len(best['PM2.5'].values)),best['PM2.5'].values,label='submit_file_8.05')
# plt.legend()
# plt.show()

# best['PM2.5'] = np.round(best['PM2.5'].values/0.004)*0.004
# best.to_csv('./_save/finedust/submit_8.05.csv')


import pandas as pd
import numpy as np

# 예시 데이터 프레임 생성
df = pd.DataFrame({'date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06'],
                   'value': [2, 5, 8, 7, 10, 13]})

# 이동평균 계산을 위한 윈도우 사이즈 설정
window_size = 3

# 이동평균을 계산하여 새로운 칼럼에 저장
df['smoothed'] = df['value'].rolling(window_size, center=True).mean()

# 결과 출력
print(df)