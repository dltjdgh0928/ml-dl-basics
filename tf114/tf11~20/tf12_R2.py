import tensorflow as tf

x_train = [1, 2, 3]
y_train = [1, 2, 3]
x_test = [4, 5, 6]
y_test = [4, 5, 6]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

lr = 0.1
# gradient = tf.reduce_mean((x*w-y)* x)
gradient = tf.reduce_mean((hypothesis-y)* x)

# y = x*w + b
# w_f = w_i - lr*(delta e / delta w)


descent = w - lr * gradient
update = w.assign(descent)          # w = w - lr * gradient


w_history = []
loss_history = []
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={x:x_train, y:y_train})
    print(step, '\t', loss_v, '\t', w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)
sess.close()

print('w_history : ', w_history)
print('loss_history : ', loss_history)

from sklearn.metrics import r2_score, mean_absolute_error

y_pred = x_test * w_v

print('r2 : ', r2_score(y_test, y_pred))
print('mae : ', mean_absolute_error(y_test, y_pred))
