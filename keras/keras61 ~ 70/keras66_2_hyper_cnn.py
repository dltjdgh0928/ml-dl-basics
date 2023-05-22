import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout, LSTM
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1).astype('float32')/255.


# 2. 모델
def build_model(dr=0.5, opt='adam', act='relu', node1=64, node2=64, node3=64, node4=64, lr=0.001):
    inputs = Input(shape=(28, 28, 1), name='input')
    x = Conv2D(node1, 2, activation=act, name='Conv2d')(inputs)
    x = Flatten()(x)
    x = Dense(node1, activation=act, name='hidden1')(x)
    x = Dropout(dr)(x)
    x = Dense(node2, activation=act, name='hidden2')(x)
    x = Dropout(dr)(x)
    x = Dense(node3, activation=act, name='hidden3')(x)
    x = Dropout(dr)(x)
    x = Dense(node4, activation=act, name='hidden4')(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)

    model = Model(inputs = inputs, outputs = outputs)
    
    model.compile(optimizer='adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')
    return model



def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    lr = [0.001, 0.005, 0.01]
    optimizers = ['Adam', 'RMSprop', 'Adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    epochs = [5, 10, 15, 20, 25, 30]
    return {'opt' : optimizers,
            'lr' : lr,
            'dr' : dropouts,
            'act' : activations,
            'epochs' : epochs,
            'batch_size' : batchs}

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import time

es = EarlyStopping(monitor='accuracy', patience=5, verbose=1, mode='auto')

start = time.time()
# model = GridSearchCV(estimator=KerasClassifier(build_fn=build_model), param_grid=create_hyperparameter(), cv=2)
model = RandomizedSearchCV(estimator=KerasClassifier(build_fn=build_model), param_distributions=create_hyperparameter(), cv=2, n_iter=1, verbose=1)
model.fit(x_train, y_train, callbacks=[es])
end = time.time()

print('걸린시간 : ', end - start)
print('model.best_params_ : ', model.best_params_)
print('model.best_estimator : ', model.best_estimator_)
print('model.best_score_ : ', model.best_score_)
print('model.score : ', model.score(x_test, y_test))

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('acc : ', accuracy_score(y_test, y_predict))

