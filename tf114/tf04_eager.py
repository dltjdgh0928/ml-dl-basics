import tensorflow as tf
print(tf.__version__)

# 즉시실행모드
print(tf.executing_eagerly())       # False 
tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())       # False 
# tf.compat.v1.enable_eager_execution()

aaa = tf.constant('hello world')

sess = tf.compat.v1.Session()
print(sess.run(aaa))
print(aaa)