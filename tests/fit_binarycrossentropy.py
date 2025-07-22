# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pickle
from sys import argv

import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

model = keras.models.Sequential([
    keras.layers.Dense(128,input_shape=[2],
        activation='sigmoid'),
    keras.layers.Dense(1,
        activation='sigmoid'),
])
model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer='sgd',
    metrics=['accuracy'],
)
model.summary()
## training greater or less
x = np.array([[1., 3.], [1., 4.], [2., 4.], [3., 1.], [4., 1.], [4., 2.]])
t = np.array([[0.],     [0.],     [0.],     [1.],     [1.],     [1.]])
v_x = np.array([[5, 1], [1, 5], [2, 6], [6, 1], [1, 7], [7, 2]])
v_t = np.array([[1],    [0],    [0],    [1],    [0],    [1]])

history = model.fit(x,t,
    epochs=100,batch_size=16,validation_data=(v_x,v_t),
    verbose=0)

plt.plot(np.array(history.history['accuracy']),label='accuracy')
plt.plot(np.array(history.history['val_accuracy']),label='val_accuracy')
plt.plot(np.array(history.history['loss']),label='loss')
plt.plot(np.array(history.history['val_loss']),label='val_loss')
plt.legend()
plt.title('Binary cross entropy')
plt.show()
