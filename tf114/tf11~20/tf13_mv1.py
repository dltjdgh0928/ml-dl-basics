import tensorflow as tf
tf.compat.v1.set_random_seed(123)


x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

# [실습]

x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))

hypothesis = x1*w1 + x2*w2 + x3*w3 + b

loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
train = optimizer.minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    epochs = 2001
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, w1_val, w2_val, w3_val, b_val = sess.run([train, loss, w1, w2, w3, b], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
        if step %20 == 0:
            print(step, loss_val, w1_val, w2_val, w3_val, b_val)
sess.close()

y_pred = x1_data*w1_val + x2_data*w2_val + x3_data*w3_val + b_val

from sklearn.metrics import r2_score, mean_absolute_error
print('r2 : ', r2_score(y_data, y_pred))