import autokeras as ak
import time
import autokeras as ak
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
import tensorflow as tf
from keras.utils.np_utils import to_categorical

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 모델
# 3. 컴파일, 훈련
path = './_save/autokeras/'
model = tf.keras.models.load_model(path + 'keras62_autokeras.h5')

# 4. 평가, 예측
y_test = to_categorical(y_test)
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print('결과 : ', results)
