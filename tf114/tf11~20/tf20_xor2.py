import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
tf.set_random_seed(337)
tv = tf.compat.v1

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
xp = tv.placeholder(tf.float32, shape=[None, 2])
yp = tv.placeholder(tf.float32, shape=[None, 1])

node_1 = 100
node_2 = 70
# model.add(Dense(10, input_shape=(2,)))
w1 = tv.Variable(tf.random.normal([x_data.shape[1], node_1]), name='weight1')
b1 = tv.Variable(tf.zeros([node_1]), name='bias1')
layer1 = tv.matmul(xp, w1) + b1

# model.add(Dense(7))
w2 = tv.Variable(tv.random.normal([node_1, node_2]), name='weight2')
b2 = tv.Variable(tv.zeros([node_2]), name='bias2')
layer2 = tv.sigmoid(tv.matmul(layer1, w2) + b2)

# model.add(Dense(1, activation='sigmoid'))
w3 = tv.Variable(tv.random.normal([node_2, 1]), name='weight3')
b3 = tv.Variable(tv.zeros([1]), name='bias3')
hypothesis = tv.sigmoid(tv.matmul(layer2, w3) + b3)




# 2. 모델
# hypothesis = tv.sigmoid(tv.matmul(xp, w) + b)


# 3-1 컴파일
cost = -tf.reduce_mean(yp*tf.log1p(hypothesis) + (1-yp) * tf.log1p(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, yp), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(1000):
        cost_val, _ = sess.run([cost, train], feed_dict = {xp:x_data, yp:y_data})

        if step & 200 == 0:
            print(step, cost_val)
            
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {xp:x_data, yp:y_data})
    print(f'hypothesis : \n{h}\n predicted : \n{p}\n accuracy : {a}')