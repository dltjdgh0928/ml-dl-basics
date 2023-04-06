# import re
# import numpy as np
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# path = './_data/pp/homo_sapiens/'

# with open(path + '1.txt', 'r') as f:
#     data = f.read().replace('\n', '').replace(' ', '')      # 공백제거
# data = re.sub(r"[0-9]", "", data)       # 라벨링 숫자 제거
# data = data.replace('a', '1')
# data = data.replace('t', '2')
# data = data.replace('c', '3')
# data = data.replace('g', '4')
# data = np.array([int(i) for i in data])
# print(data)
# print(data.shape)
# data = data.reshape(1, -1)
# print(data)
# # data = [[i] for i in data]
# data = pad_sequences(data, maxlen=700, padding='pre', truncating='pre')
# print(data)
# print(data.shape)



# import os
# import re
# import numpy as np
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# path = './_data/pp/homo_sapiens/'

# for filename in os.listdir(path):
#     if filename.endswith('.txt'):
#         with open(os.path.join(path, filename), 'r') as f:
#             data = f.read().replace('\n', '').replace(' ', '')      # 공백제거
#         data = re.sub(r"[0-9]", "", data)       # 라벨링 숫자 제거
#         data = data.replace('a', '1')
#         data = data.replace('t', '2')
#         data = data.replace('c', '3')
#         data = data.replace('g', '4')
#         data = np.array([int(i) for i in data])
#         data = data.reshape(1, -1)
#         data = pad_sequences(data, maxlen=700, padding='pre', truncating='pre')
#         data = data.T
#         print(f"{filename}: {data.shape}")

# print(data.shape)


import os
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

path = './_data/pp/homo_sapiens/'

data_list = []  # 각 데이터를 저장할 리스트

for filename in os.listdir(path):
    if filename.endswith('.txt'):
        with open(os.path.join(path, filename), 'r') as f:
            data = f.read().replace('\n', '').replace(' ', '')      # 공백제거
        data = re.sub(r"[0-9]", "", data)       # 라벨링 숫자 제거
        data = data.replace('a', '1').replace('t', '2').replace('c', '3').replace('g', '4')
        data = np.array([int(i) for i in data])
        data = data.reshape(1, -1)
        data = pad_sequences(data, maxlen=700, padding='pre', truncating='pre')
        data = data.T
        
        data_list.append(data)  # 처리된 데이터를 리스트에 추가

all_data = np.concatenate(data_list, axis=1)  # 리스트에 있는 모든 데이터를 합침
all_data = all_data.reshape(all_data.shape[1], all_data.shape[0], 1)  # shape을 변경함
print(all_data.shape)  # (25, 700, 1)

