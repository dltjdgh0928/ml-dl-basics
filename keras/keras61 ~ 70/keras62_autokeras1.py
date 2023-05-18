import time
import autokeras as ak
from sklearn.model_selection import train_test_split
from keras.datasets import mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 모델
model = ak.ImageClassifier(
    overwrite=False,
    max_trials=2
)

# 3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train, epochs=2, validation_split=0.15)
end = time.time()

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print('결과 : ', results)
print('걸린시간 : ', round(end-start, 4))

best_model = model.export_model()   
print(best_model.summary())

path = './_save/autokeras/'
best_model.save(path + 'kraes62_autokeras.h5')
