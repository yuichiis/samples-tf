# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import ndarray
#import matplotlib.pyplot as plt

x = tf.Variable([1., 2., 98., 1., 1., 99., 3., 1., 3., 96., 4., 1.],dtype=tf.dtypes.float32)

z = tf.math.top_k(
    x,
    k=3,
    sorted=False,
    index_type=tf.dtypes.int64
)
print('x=',x)
print('values=',z.values.numpy())
print('indices=',z.indices.numpy())

