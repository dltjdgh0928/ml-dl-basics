import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(337)

# 1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

# 2. 모델
model = Sequential()
model.add(Dense(2, input_shape=(1,)))
model.add(Dense(1))
model.summary()

print(model.weights)
print('================================================')
print(model.trainable_weights)

print(len(model.weights))
print(len(model.trainable_weights))

model.trainable = True
# model.trainable = False

print(len(model.weights))
print(len(model.trainable_weights))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.001))

model.fit(x, y, batch_size=1, epochs=1)

y_predict = model.predict(x)
print(y_predict)

print(model.weights)
print('===================')
print(model.trainable_weights)
