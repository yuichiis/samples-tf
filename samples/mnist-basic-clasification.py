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

if len(argv)>1 and int(argv[1]):
    shrink = True
else:
    shrink = False

dataset='mnist'
if len(argv)>2 and argv[2]:
    dataset=argv[2]

if dataset=='fashion':
    (train_img,train_label),(test_img,test_label) = \
        keras.datasets.fashion_mnist.load_data()
    inputShape = [28*28]
    shrinkEpochs = 3
    shrinkTrainSize = 5000
    shrinkTestSize  = 100
elif dataset=='cifar10':
    (train_img,train_label),(test_img,test_label) = \
        keras.datasets.cifar10.load_data()
    inputShape = [32*32*3]
    shrinkEpochs = 3
    shrinkTrainSize = 2000
    shrinkTestSize  = 100
else:
    (train_img,train_label),(test_img,test_label) = \
        keras.datasets.mnist.load_data()
    inputShape = [28*28]
    shrinkEpochs = 3
    shrinkTrainSize = 5000
    shrinkTestSize  = 100


print("train=["+','.join([str(i) for i in train_img.shape])+"]")
print("test=["+','.join([str(i) for i in test_img.shape])+"]")
print("train_label=["+','.join([str(i) for i in train_label.shape])+"]")
print("test_label=["+','.join([str(i) for i in train_label.shape])+"]")

if shrink:
    # Shrink data
    epochs = shrinkEpochs;
    trainSize = shrinkTrainSize;
    testSize  = shrinkTestSize;
    print("Shrink data ...")
    train_img = train_img[0:trainSize]
    train_label = train_label[0:trainSize]
    test_img = test_img[0:testSize]
    test_label = test_label[0:testSize]
    print("Shrink train=["+','.join([str(i) for i in train_img.shape])+"]")
    print("Shrink test=["+','.join([str(i) for i in test_img.shape])+"]")


# flatten images and normalize
def formatingImage(train_img):
    dataSize = train_img.shape[0]
    imageSize = train_img[0].size
    train_img = train_img.reshape([dataSize,imageSize])
    return 1.0 / 255.0 * train_img.astype(np.float32)


print("formating train images ...")
train_img = formatingImage(train_img)
print("formating test images ...")
test_img  = formatingImage(test_img)

print("creating model ...")
model = keras.models.Sequential([
    keras.layers.Dense(units=128,
        input_shape=inputShape,
        #kernel_initializer='glorot_uniform',
        activation='relu',
    ),
    keras.layers.Dense(units=10)
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'],
)


print("training model ...")
history = model.fit(train_img,train_label,
    epochs=5,batch_size=256,validation_data=(test_img,test_label))

#model.save('mnist-basic-model.weights.h5')

#model = keras.models.load_model('mnist-basic-model.weights.h5')

plt.plot(np.array(history.history['accuracy']),label='accuracy')
plt.plot(np.array(history.history['val_accuracy']),label='val_accuracy')
plt.plot(np.array(history.history['loss']),label='loss')
plt.plot(np.array(history.history['val_loss']),label='val_loss')
plt.legend();
plt.title(dataset)
plt.show()
