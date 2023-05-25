from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')
# model = ResNet50(weights=None)
# model = ResNet50(weights='경로')

# path = f'c:/_data/cat_dog/train/dog.6.jpg'
path = f'c:/_data/suit/뷔.jfif'

img = image.load_img(path, target_size=(224, 224))
# print(img)

x = image.img_to_array(img)
# print(x, '\n', x.shape)
# print(np.min(x), np.max(x))

# x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
# x = x.reshape(1, *x.shape)
# print(x.shape)

x = np.expand_dims(x, axis=0)

# -1 에서 1 사이로 정규화
x = preprocess_input(x)
print(x.shape)
# print(np.min(x), np.max(x))

x_pred = model.predict(x)
print(x_pred, '\n', x_pred.shape)

print('결과 : ', decode_predictions(x_pred, top=5))
