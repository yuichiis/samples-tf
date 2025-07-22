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

print('tensorflow: %s, keras: %s' % (tf.__version__, keras.__version__))

model = keras.models.Sequential([
    keras.layers.Dense(
        16,# units
        input_shape=(10,),
        #activation='sigmoid',
        activation='tanh',
    ),
    #keras.layers.Dense(
    #    10,# units
    #),
    keras.layers.Activation('softmax')
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer='sgd',
    metrics=['accuracy'],
)
model.summary()

x = np.array([
    [0],
    [9],
    [1],
    [5],
],np.int32)
t = np.array(
    [0,9,1,5]
)
v_x = np.array([
    [2],
    [1],
    [4],
    [9],
],np.int32)
v_t = np.array(
    [2,1,4,9]
)
x = keras.utils.to_categorical(x.reshape(4,), num_classes=10).reshape(4,10)
v_x = keras.utils.to_categorical(v_x.reshape(4,), num_classes=10).reshape(4,10)

print('x=',x.shape)
print('t=',t.shape)

print("training model ...")
history = model.fit(x,t,
    epochs=300,batch_size=1,verbose=0,
    validation_data=(v_x,v_t))

plt.plot(np.array(history.history['accuracy']),label='accuracy')
plt.plot(np.array(history.history['val_accuracy']),label='val_accuracy')
plt.plot(np.array(history.history['loss']),label='loss')
plt.plot(np.array(history.history['val_loss']),label='val_loss')
plt.legend()
plt.title('SimSingleRNN')
plt.show()
