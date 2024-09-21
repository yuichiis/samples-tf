# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

x = tf.Variable([1.0,2.0])
y = tf.Variable([3.0,2.0])

with tf.GradientTape() as tape:
    z = tf.math.equal(
        x,y
    )
    z = tf.cast(z,tf.float32)
print('x=',x)
print('y=',y)
print('z=',z)
grads = tape.gradient(z,[x,y])
print('dx=',grads[0])
print('dy=',grads[1])

