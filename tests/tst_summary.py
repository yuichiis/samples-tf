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
import copy

class TestModule(keras.Model):
    '''
    test
    '''
    def __init__(
        self,
        **kwargs
    ):
        '''test'''
        super(TestModule, self).__init__(**kwargs)
        self.dense2 = keras.layers.Dense(2,name='firstDense')
        self.dense1 = keras.layers.Dense(100,)#,input_shape=(2,))

    def call(self,inputs):
        x = self.dense1(inputs)
        outputs = self.dense2(x)
        return outputs

inputs=np.array([[2.,3.]],dtype=np.float32)
trues=np.array([1])

module = TestModule()
#module.compile(
#    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#    optimizer=keras.optimizers.Adam())
#module.fit(inputs,trues,epochs=1)
module.build(input_shape=(1,2,))
module.summary()

module2 = keras.Sequential([
    keras.layers.Dense(100,input_shape=(2,)),
    keras.layers.Dense(2,)
])
module2.summary()

dense = keras.layers.Dense(2)
print(dense.name)
dense2 = copy.deepcopy(dense)
print(dense2.name)
dense3 = keras.layers.Dense(2)
print(dense3.name)
dense4 = keras.layers.Dense(2,name='firstDense')
print(dense4.name)
