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

class CustomCallback(keras.callbacks.Callback):
    def __init__(self,**kwargs):
        super(CustomCallback, self).__init__()
        self.gradlog = {}
        self.prev_w = None
    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.get_weights();
        if self.prev_w is None:
            self.prev_w = weights
            return
        num = 0;
        next = []
        for w,prev in zip(weights,self.prev_w):
            name = 'g'+str(num)+'('+str(w.shape)+')'
            if self.gradlog.get(name) is None:
                self.gradlog[name] = [];
            g = prev - w
            self.gradlog[name].append(np.max(np.abs(g)))
            #echo sprintf("%7.2f",strval())." ";
            num += 1
            next.append(w.copy())
        self.prev_w = next

model = keras.models.Sequential([
    keras.layers.GRU(
        10,# units
        input_shape=(4,10),
        #input_shape=(1,10),
        return_sequences=True,
    ),
    keras.layers.Activation('softmax')
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer='adam',
    #optimizer='sgd',
    metrics=['accuracy'],
)
model.summary()
for w in model.get_weights():
    print(np.amax(w))

#x = np.array([
#    [0],
#    [9],
#    [1],
#    [5],
#],np.int32)
#t = np.array(
#    [0,9,1,5]
#)
#v_x = np.array([
#    [2],
#    [1],
#    [4],
#    [9],
#],np.int32)
#v_t = np.array(
#    [0,9,1,5]
#)
#x = keras.utils.to_categorical(x.reshape(4,), num_classes=10).reshape(4,1,10)
#v_x = keras.utils.to_categorical(v_x.reshape(4,), num_classes=10).reshape(4,1,10)

x = np.array([
    [0,1,2,9],
    [9,8,7,6],
    [1,3,3,4],
    [5,4,3,2],
],np.int32)
t = np.array([
    [0,1,2,9],
    [9,8,7,6],
    [1,3,3,4],
    [5,4,3,2],
])
v_x = np.array([
    [2,3,3,4],
    [1,1,1,4],
    [4,3,3,1],
    [9,3,3,2],
],np.int32)
v_t = np.array([
    [2,3,3,4],
    [1,1,1,4],
    [4,3,3,1],
    [9,3,3,2],
])
x = keras.utils.to_categorical(x.reshape(16,), num_classes=10).reshape(4,4,10)
v_x = keras.utils.to_categorical(v_x.reshape(16,), num_classes=10).reshape(4,4,10)

print('x=',x.shape)
print('t=',t.shape)

callback = CustomCallback()

print("training model ...")
history = model.fit(x,t,
    epochs=300,batch_size=1,verbose=0,
    validation_data=(v_x,v_t),
    callbacks=[callback])

plt.plot(np.array(history.history['accuracy']),label='accuracy')
plt.plot(np.array(history.history['val_accuracy']),label='val_accuracy')
plt.plot(np.array(history.history['loss']),label='loss')
plt.plot(np.array(history.history['val_loss']),label='val_loss')
plt.legend()
plt.title('GRU return_sequences')
plt.figure()
for key,gradlog in callback.gradlog.items():
    plt.plot(np.array(gradlog),label=key)
plt.legend()
plt.show()
