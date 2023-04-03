from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1. 데이터
# 1.1 경로, 가져오기
path = 'd:/study_data/_data/air/dataset/'
path_save = 'd:/study_data/_save/air/'

train_csv = pd.read_csv(path + 'train_data.csv')
test_csv = pd.read_csv(path + 'test_data.csv')

# 1.2 확인사항
# print(type(dataset))                    # <class 'pandas.core.frame.DataFrame'>
# print(dataset.shape)                    # (2463, 8)
# print(dataset.columns)                  # Index(['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current',
#     #    'motor_rpm', 'motor_temp', 'motor_vibe', 'type'],
#     #   dtype='object')
# print(dataset.isnull().sum())           # 결측치 x
# print(dataset.info())
# print(dataset.describe())

# 1.3 x, y 분리
x = train_csv.drop(['out_pressure', 'type'], axis=1)
y = train_csv['type']
test_csv = test_csv.drop(['out_pressure', 'type'], axis=1)

# 1.4 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)

print(x_train.shape)


# 1.5 scaler 
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = Sequential()
model.add(Dense(64, input_shape=(6,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='softmax'))