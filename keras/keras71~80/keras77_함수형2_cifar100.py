import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(x_train, y_train), (x_test, y_test) = cifar100.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

vgg16_f = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

result_list = []
test_acc_list = []

for i in range(4):
    if 0 <= i < 2:
        input1 = vgg16_f
    elif 2 <= i:
        vgg16_f.trainable = False
        input1 = vgg16_f

    if i == 0 or i == 2:
        flat = Flatten()(input1.output)
    
    elif i == 1 or i == 3:
        flat = GlobalAveragePooling2D()(input1.output)
    
    dense2 = Dense(100)(flat)
    dense3 = Dense(100)(dense2) 
    output = Dense(100, activation='softmax')(dense3)
    
    model = Model(inputs=input1.input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    result = model.evaluate(x_test, y_test)
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1))
    
    print(f'{i+1} result : {result}\nacc : {acc}')
    result_list.append(result[0])
    test_acc_list.append(acc)

print(f'{result_list}\n{test_acc_list}')