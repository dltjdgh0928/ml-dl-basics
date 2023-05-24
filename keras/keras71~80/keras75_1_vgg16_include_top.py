import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
# model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

model.summary()
print(len(model.weights))
print(len(model.trainable_weights))

# include_top = True

# 1. FC layer 원래꺼 쓴다
# 2. input_shape = (224, 224, 3) 고정

#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0

#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792
 
#  ...
 
#  flatten (Flatten)           (None, 25088)             0

#  fc1 (Dense)                 (None, 4096)              102764544

#  fc2 (Dense)                 (None, 4096)              16781312

#  predictions (Dense)         (None, 1000)              4097000



# include_top = False
# 1. FC layer 원래거 삭제
# 2. input_shape = (32, 32, 3) -> 커스터마이징 가능

#  input_1 (InputLayer)        [(None, 32, 32, 3)]       0

#  block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792
 
#  ...
 
#  block5_conv1 (Conv2D)       (None, 2, 2, 512)         2359808

#  block5_conv2 (Conv2D)       (None, 2, 2, 512)         2359808

#  block5_conv3 (Conv2D)       (None, 2, 2, 512)         2359808

#  block5_pool (MaxPooling2D)  (None, 1, 1, 512)         0