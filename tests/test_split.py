import tensorflow as tf
import numpy as np

states = tf.Variable([
    [1., 2., 3., 4., 5., 6.],
    [7., 8., 9., 10., 11., 12.],
])

with tf.GradientTape() as tape:
    mean, logstd = tf.split(states, 2, axis=-1) # Split x into 2 parts 

print('states:',states.shape) # (5,)
print(states)
print('mean:',mean.shape) # (5,)
print(mean) 
print('logstd:',logstd.shape) # (5,)
print(logstd) 
# states: (2, 6)
# <tf.Variable 'Variable:0' shape=(2, 6) dtype=float32, numpy=
# array([[ 1.,  2.,  3.,  4.,  5.,  6.],
#        [ 7.,  8.,  9., 10., 11., 12.]], dtype=float32)>
# mean: (2, 3)
# tf.Tensor(
# [[1. 2. 3.]
#  [7. 8. 9.]], shape=(2, 3), dtype=float32)
# logstd: (2, 3)
# tf.Tensor(
# [[ 4.  5.  6.]
#  [10. 11. 12.]], shape=(2, 3), dtype=float32)

dstatus_dlogstd = tape.gradient(logstd, states)
print('dstatus_dlogstd', dstatus_dlogstd)
# dstatus_dlogstd tf.Tensor(
# [[0. 0. 0. 1. 1. 1.]
#  [0. 0. 0. 1. 1. 1.]], shape=(2, 6), dtype=float32)

