import tensorflow as tf

tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()

print(tf.__version__)
print('즉시 실행 모드 : ', tf.executing_eagerly())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(gpus[0])
    except RuntimeError as e:
        print(e)        
else:
    print('gpu 없다')
    
    
