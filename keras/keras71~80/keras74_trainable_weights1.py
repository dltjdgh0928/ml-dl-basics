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
model.add(Dense(3, input_shape=(1,)))
model.add(Dense(2))
model.add(Dense(1))

model.summary()


print(model.weights)
print('================================================')
print(model.trainable_weights)

print(len(model.weights))
print(len(model.trainable_weights))

print(model.weights[0])
print(model.weights[1])
print(model.weights[2])
print(model.weights[3])
print(model.weights[4])
print(model.weights[5])

model.trainable = False

print(len(model.weights))
print(len(model.trainable_weights))