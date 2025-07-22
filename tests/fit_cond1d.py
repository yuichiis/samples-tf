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
    keras.layers.Conv1D(
        128,# filters
        3,# kernel_size
        input_shape=(10,1),
        kernel_initializer='he_normal',
        #kernel_initializer='random_normal',
        activation='relu',
    ),
    keras.layers.MaxPooling1D(),
    keras.layers.Conv1D(128,3),#activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Activation('softmax')
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer='adam',
    metrics=['accuracy'],
)
model.summary()

x = np.array([
    [[0.1],[0.1],[0.2],[0.2],[0.3],[0.3],[0.4],[0.4],[0.5],[0.5]],
    [[0.9],[0.9],[0.8],[0.8],[0.7],[0.7],[0.6],[0.6],[0.5],[0.5]],
    [[0.5],[0.5],[0.6],[0.6],[0.7],[0.7],[0.8],[0.8],[0.9],[0.9]],
    [[0.5],[0.5],[0.4],[0.4],[0.3],[0.3],[0.2],[0.2],[0.1],[0.1]],
])
t = np.array(
    [1,0,1,0]
)
v_x = np.array([
    [[0.1],[0.1],[0.25],[0.25],[0.35],[0.35],[0.45],[0.45],[0.6], [0.6] ],
    [[0.9],[0.9],[0.7], [0.7], [0.5], [0.5], [0.3], [0.3], [0.1], [0.1] ],
    [[0.1],[0.1],[0.11],[0.11],[0.12],[0.12],[0.13],[0.13],[0.14],[0.14]],
    [[0.5],[0.5],[0.45],[0.45],[0.4], [0.4], [0.35],[0.35],[0.3], [0.3] ],
])
v_t = np.array(
    [1,0,1,0]
)

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
plt.legend();
plt.title('conv1d')
plt.show()
