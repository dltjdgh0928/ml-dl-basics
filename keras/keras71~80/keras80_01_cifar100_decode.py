from tensorflow.keras.applications import VGG19, Xception, ResNet50, ResNet101, InceptionV3, InceptionResNetV2, DenseNet121, MobileNetV2, NASNetMobile, EfficientNetB0
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D
import numpy as np

model_list = [VGG19, Xception, ResNet50, ResNet101, InceptionV3, InceptionResNetV2, DenseNet121, MobileNetV2, NASNetMobile, EfficientNetB0]
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x = preprocess_input(x_train[0])
x = x.reshape(1, *x.shape)

for i in range(len(model_list)):
    input1 = model_list[i](weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    flat = GlobalAveragePooling2D()(input1.output)
    output = Dense(100, activation='softmax')(flat)
    model = Model(inputs=input1.input, outputs=output)
    
    x_pred = model.predict(x)
    x_pred_batch = np.repeat(x_pred, 10, axis=-1)
    print('result : ', decode_predictions(x_pred_batch, top=5))