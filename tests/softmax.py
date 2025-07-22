import tensorflow as tf
from functools import reduce
from operator import mul

#x = tf.constant([
#    [[[1., 2., 3., 4.],
#      [1., 2., 3., 4.],
#      [1., 2., 3., 4.]],
#     [[1., 2., 3., 4.],
#      [1., 2., 3., 4.],
#      [1., 2., 3., 4.]]],
#    [[[1., 2., 3., 4.],
#      [1., 2., 3., 4.],
#      [1., 2., 3., 4.]],
#     [[1., 2., 3., 4.],
#      [1., 2., 3., 4.],
#      [1., 2., 3., 4.]]],
#])
shape = [2, 8, 6, 7]
x = 512.0 * tf.ones(shape)
x = tf.Variable(x)
saltmax = reduce(mul, shape)
#salt = tf.reshape(tf.range(0, saltmax, dtype=tf.float32)/saltmax, shape)
#salt = tf.reshape(tf.range(0, reduce(mul, x.shape), dtype=tf.float32), x.shape)
salt = tf.reshape(tf.range(0, reduce(mul, shape), dtype=tf.float32),shape)

#x = tf.Variable(tf.reshape(x, [2*8*6, 7]))
#salt = tf.reshape(salt, x.shape)

#shape = [3,3]
#salt = tf.reshape(tf.range(0, reduce(mul, shape), dtype=tf.float32),shape)
#x = tf.Variable(1/(1+salt))


def custom_softmax_log_softmax(x):
    return tf.exp(tf.nn.log_softmax(x, axis=-1))

def custom_softmax_clip(x):
    submax = x - tf.reduce_max(x, axis=-1, keepdims=True)
    exp_x = tf.exp(tf.clip_by_value(submax, -100.0, 100.0)) # 値をクリップ
    sumexp = tf.reduce_sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sumexp

def custom_softmax(x) :
    submax = x - tf.reduce_max(x, axis=-1, keepdims=True)
    exp_x = tf.exp(submax)
    sumexp = tf.reduce_sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sumexp

print('===== inputs =====')
print('x',x)
print('===== softmax =====')
with tf.GradientTape() as tape:
    softmax = tf.nn.softmax(x,axis=-1)
    #softmax = custom_softmax(x)
    #softmax = custom_softmax_log_softmax(x)
    #softmax = custom_softmax_clip(x)
    y = salt * softmax
print('softmax',softmax)

grads = tape.gradient(y, [x, softmax])
print('===== grads =====')
print('d_inputs',grads[1])
print('dx',grads[0])
