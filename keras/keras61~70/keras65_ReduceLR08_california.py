import numpy as np
from sklearn.datasets import fetch_california_housing


datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)

# 2. 모델 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim = 8))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

rl = ReduceLROnPlateau(monitor='val_loss', patience=4, mode='auto', verbose=1, factor=0.5)
es = EarlyStopping(monitor='val_loss', patience=10)
optimizer = Adam(learning_rate=0.1)
model.compile(loss = 'mse', optimizer=optimizer)
model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.2, callbacks=[rl, es])

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score

print('r2 : ', r2_score(y_test, y_pred))