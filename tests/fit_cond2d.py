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
    keras.layers.Conv2D(
        128,# filters
        3,# kernel_size
        input_shape=(10,10,1),
        kernel_initializer='he_normal',
        #kernel_initializer='random_normal',
        activation='relu',
        #activation='softmax',
    ),
    keras.layers.MaxPooling2D(),
    #keras.layers.Dropout(0.25),
    keras.layers.Conv2D(128,3),#activation='relu',kernel_initializer='he_normal',),
    keras.layers.Flatten(),
    keras.layers.Activation('softmax')
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer='adam',
    metrics=['accuracy'],
)
model.summary();

x = np.zeros([4,10,10,1],dtype=np.float32)
for i in range(0,10,1):
    x[0][i][i][0]=1.0
for i in range(0,10,1):
    x[1][i][9-i][0]=1.0
for i in range(1,9,1):
    x[2][i+1][i+1][0]=1.0
for i in range(1,9,1):
    x[3][i+1][9-i][0]=1.0
t = np.array(
    [1,0,1,0]
)
v_x = np.zeros([4,10,10,1],dtype=np.float32)
for i in range(0,8,1):
    v_x[0][i][i+2][0]=1.0
for i in range(0,8,1):
    v_x[1][i][9-i][0]=1.0
for i in range(1,8,1):
    v_x[2][i+2][i+1][0]=1.0
for i in range(1,8,1):
    v_x[3][i+2][9-i][0]=1.0
v_t = np.array(
    [1,0,1,0]
);

print('x=',x.shape)
print('t=',t.shape)

print("training model ...")
history = model.fit(x,t,
    epochs=100,batch_size=1,verbose=0,
    validation_data=(v_x,v_t))

plt.plot(np.array(history.history['accuracy']),label='accuracy')
plt.plot(np.array(history.history['val_accuracy']),label='val_accuracy')
plt.plot(np.array(history.history['loss']),label='loss')
plt.plot(np.array(history.history['val_loss']),label='val_loss')
plt.legend()
plt.title('conv2d')
plt.show()
