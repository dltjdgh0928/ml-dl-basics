import tensorflow as tf
tf.set_random_seed(337)
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)


# 2. 모델 구성
hypothesis = x * w + b


# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))        # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train = optimizer.minimize(loss)

# model.compile(loss='mse', optimizer='sgd') 와 동일

# 3-2 훈련
loss_val_list = []
w_val_list = []

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    epochs = 101
    for step in range(epochs):
        # sess.run(train)
        tr, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:[1,2,3,4,5], y:[2,4,6,8,10]})
        if step %20 == 0:
            print(tr, step, loss_val, w_val, b_val)
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
    x_data = tf.placeholder(tf.float32, shape=[None])
    y_pred = x_data * w_val + b_val
    y_predict = sess.run([y_pred], feed_dict={x_data:[6,7,8]})
    print('y_predict : ', (y_predict[0]))

print(loss_val_list)
print(w_val_list[0])


import matplotlib.pyplot as plt
plt.subplot(1,3,1)
plt.plot(loss_val_list)
plt.xlabel('epochs')
plt.ylabel('loss')

plt.subplot(1,3,2)
plt.plot(w_val_list)
plt.xlabel('epochs')
plt.ylabel('weight')

plt.subplot(1,3,3)
plt.scatter(w_val_list, loss_val_list)
plt.xlabel('weight')
plt.ylabel('loss')
plt.show()