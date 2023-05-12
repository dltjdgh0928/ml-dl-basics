import tensorflow as tf
tf.compat.v1.set_random_seed(337)

# [실습]
# 08_2 
# 1. Session() // sess.run(변수)
# 1. Session() // 변수.eval(session=sess)
# 1. Session() // 변수.eval()



import tensorflow as tf
tf.set_random_seed(337)

# 1. 데이터
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)


# 2. 모델 구성
hypothesis = x * w + b


# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))        # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train = optimizer.minimize(loss)

# model.compile(loss='mse', optimizer='sgd') 와 동일

# 3-2 훈련


# 1. sess.run
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    epochs = 101
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:[1,2,3,4,5], y:[2,4,6,8,10]})
        if step %100 == 0:
            print(step, loss_val, w_val, b_val)
    x_data = tf.placeholder(tf.float32, shape=[None])
    y_pred = x_data * w_val + b_val
    y_predict = sess.run([y_pred], feed_dict={x_data:[6,7,8]})
    print('y_predict : ', (y_predict[0]))
    


# 2. 변수.eval(session=sess)
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    epochs = 101
    for step in range(epochs):
        # sess.run(train)
        _, loss_val= sess.run([train, loss], feed_dict={x:[1,2,3,4,5], y:[2,4,6,8,10]})
        if step %100 == 0:
            print(step, loss_val)
    w1 = w.eval(session=sess)
    b1 = b.eval(session=sess)
     
    x_data = tf.placeholder(tf.float32, shape=[None])
    y_pred = x_data * w1 + b1
    y_predict = sess.run([y_pred], feed_dict={x_data:[6,7,8]})
    print('y_predict : ', (y_predict[0]))



# 3. 변수.eval()
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.global_variables_initializer())

epochs = 101
for step in range(epochs):
    # sess.run(train)
    _, loss_val = sess.run([train, loss], feed_dict={x:[1,2,3,4,5], y:[2,4,6,8,10]})
    
    if step %100 == 0:
        print(step, loss_val)
w2 = w.eval()
b2 = b.eval()

x_data = tf.placeholder(tf.float32, shape=[None])
y_pred = x_data * w2 + b2
y_predict = sess.run([y_pred], feed_dict={x_data:[6,7,8]})
print('y_predict : ', (y_predict[0]))

