from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import time
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes]

for i in range(len(data_list)):
    x, y = data_list[i](return_X_y=True)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=337)
    

    # 2. 모델
    def build_model(dr=0.2, opt='adam', act='relu', node1=64, node2=64, node3=64, node4=64, lr=0.001):
        inputs = Input(shape=(x_train.shape[1]), name='input')
        x = Dense(node1, activation=act, name='hidden1')(inputs)
        x = Dropout(dr)(x)
        x = Dense(node2, activation=act, name='hidden2')(x)
        x = Dropout(dr)(x)
        x = Dense(node3, activation=act, name='hidden3')(x)
        x = Dropout(dr)(x)
        x = Dense(node4, activation=act, name='hidden4')(x)
        outputs = Dense(10, activation='softmax', name='outputs')(x)
        model = Model(inputs = inputs, outputs = outputs)
        if i < 5:
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer='adam', loss='mse')
        return model

    def create_hyperparameter():
        lr = [0.001, 0.005, 0.01]
        optimizers = [Adam(lr), RMSprop(lr), Adadelta(lr)]
        dropouts = [0.2, 0.3, 0.4, 0.5]
        activations = ['relu', 'elu', 'selu', 'linear']
        batchs = [100, 200, 300, 400, 500]
        epochs = [5, 10, 15, 20]
        return {'dr' : dropouts, 'opt' : optimizers, 'act' : activations, 'lr' : lr, 'epochs' : epochs, 'batch_size' : batchs}

    start = time.time()
    model = RandomizedSearchCV(estimator=KerasClassifier(build_fn=build_model), param_distributions=create_hyperparameter(), cv=2, n_iter=1, verbose=1)
    model.fit(x_train, y_train, verbose=0)
    end = time.time()

    best_params_ = model.best_params_.copy()
    best_params_['opt'] = model.best_params_['opt'].__class__.__name__
    
    print(f'\n{data_list[i].__name__}\ntime used : {end - start}\nmodel.best_params_ : {best_params_}')    
    if i < 5:
        print(f'acc : {accuracy_score(y_test, model.predict(x_test))}\n')
    else:
        print(f'r2 : {r2_score(y_test, model.predict(x_test))}\n')