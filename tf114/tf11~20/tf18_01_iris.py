import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype
from sklearn.metrics import r2_score, accuracy_score
tf.compat.v1.set_random_seed(337)

tf.compat.v1.disable_eager_execution()

data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype]
learning_rate_list = [0.1, 1000, 1000, 0.01, 0.1]
import time
start = time.time()

for i in range(len(data_list)):
    tf.compat.v1.global_variables_initializer()
    x_data, y_data = data_list[i](return_X_y=True)
    if i == 4:
        y_data = y_data - 1
    y_col_num = len(np.unique(y_data))
    y_data = tf.keras.utils.to_categorical(y_data)
    # y_data = tf.one_hot(y_data, depth=len(np.unique(y_data)), axis=1)

    # 2. 모델구성
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_data.shape[1]])
    w = tf.compat.v1.Variable(tf.random.normal([x_data.shape[1], y_col_num]), name='weight')
    b = tf.compat.v1.Variable(tf.zeros([1, y_col_num]), name='bias')
    y = tf.compat.v1.placeholder(tf.float32, shape=[None, y_col_num])

    hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)

    # 3-1 컴파일
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=hypothesis))
    # loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
    train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate_list[i]).minimize(loss)

    # 3-2. 훈련
    epochs = 3000
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for step in range(epochs):            
            _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})

        # 4. 평가, 예측
        xp2 = tf.compat.v1.placeholder(tf.float32, shape=[None, x_data.shape[1]])
        y_pred = tf.nn.softmax(tf.compat.v1.matmul(xp2, w_val) + b_val)
        y_predict = sess.run([y_pred], feed_dict={xp2:x_data})
        print(np.argmax(y_predict[0], axis=1))
        print(data_list[i].__name__, 'acc : ', accuracy_score(np.argmax(y_data, axis=1), np.argmax(y_predict[0], axis=1)))

end = time.time()

print(end - start)
# 274gpu 123.8초
# 114cpu 182.8초

