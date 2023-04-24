from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
n_c_list = [154, 331, 486, 713]
pca_list = [0.95, 0.99, 0.999, 1.0]

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)
y = to_categorical(y)
x = x.reshape(x.shape[0], -1)


for i in range(len(n_c_list)):
    pca = PCA(n_components=n_c_list[i])
    x_p = pca.fit_transform(x.astype('float32'))
    x_train, x_test, y_train, y_test = train_test_split(x_p, y, train_size=0.8, shuffle=True, random_state=123)

    model = Sequential()
    model.add(Dense(16, input_shape=(n_c_list[i],)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
    model.fit(x_train, y_train, epochs=1, batch_size=128, validation_split=0.2)
    
    acc = model.evaluate(x_test, y_test)[1]
    print(f'PCA {pca_list[i]} test acc : , {acc}')
    
    y_pred = np.argmax(model.predict(x_test), axis=0)
    print(f'PCA {pca_list[i]} pred acc : ,', accuracy_score(np.argmax(y_test, axis=0), y_pred))
    
# 최고 CNN :
# 최고 DNN :
# PCA 0.95 : 
# PCA 0.99 :  
# PCA 0.999 : 
# PCA 1.0 : 