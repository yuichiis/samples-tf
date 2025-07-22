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

dataset='cifar10'
epochs=10
batch_size=64

if dataset=='fashion':
    (train_img,train_label),(test_img,test_label) = \
        keras.datasets.fashion_mnist.load_data()
    inputShape = [28,28,1]
elif dataset=='cifar10':
    (train_img,train_label),(test_img,test_label) = \
        keras.datasets.cifar10.load_data()
    inputShape = [32,32,3]
else:
    (train_img,train_label),(test_img,test_label) = \
        keras.datasets.mnist.load_data()
    inputShape = [28,28,1]

print("train=["+','.join([str(i) for i in train_img.shape])+"]")
print("test=["+','.join([str(i) for i in test_img.shape])+"]")


# flatten images and normalize
def formatingImage(train_img,inputShape):
    dataSize = train_img.shape[0]
    train_img = train_img.reshape([dataSize]+inputShape)
    return 1.0 / 255.0 * train_img.astype(np.float32)


print("formating train images ...")
train_img = formatingImage(train_img,inputShape)
print("formating test images ...")
test_img  = formatingImage(test_img,inputShape)

print("creating model ...")
model = keras.models.Sequential([
    keras.layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        padding="same",
        input_shape=inputShape),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        padding="same",
        activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(
        filters=128,
        kernel_size=(3,3),
        padding="same",
        activation='relu'),
    keras.layers.Conv2D(
        filters=128,
        kernel_size=(3,3),
        padding="same",
        activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(
        filters=256,
        kernel_size=(3,3),
        padding="same",
        activation='relu'),
    keras.layers.Conv2D(
        filters=256,
        kernel_size=(3,3),
        padding="same",
        activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(
        filters=512,
        kernel_size=(3,3),
        padding="same",
        activation='relu'),
    keras.layers.Conv2D(
        filters=512,
        kernel_size=(3,3),
        padding="same",
        activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(units=1024,
        activation='relu'),
    keras.layers.Dense(units=512),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(units=10),
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])
model.summary()

print("training model ...")
history = model.fit(train_img,train_label,
    epochs=epochs,batch_size=batch_size,validation_data=(test_img,test_label))

#model.save('mnist-basic-model.weights.h5')

#model = keras.models.load_model('mnist-basic-model.weights.h5')

plt.plot(np.array(history.history['accuracy']),label='accuracy')
plt.plot(np.array(history.history['val_accuracy']),label='val_accuracy')
plt.plot(np.array(history.history['loss']),label='loss')
plt.plot(np.array(history.history['val_loss']),label='val_loss')
plt.legend();
plt.title(dataset)
plt.show()
