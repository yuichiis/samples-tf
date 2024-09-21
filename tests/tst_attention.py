# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

attention = tf.keras.layers.Attention()
#query = tf.constant([
#    [[0.0,0.5,1.0],[1.5,2.0,2.5]],
#    [[3.0,3.5,4.0],[4.5,5.0,5.5]]
#])
#value = tf.constant([
#    [[0.0,0.5,1.0],[0.0,0.5,1.0],[1.5,2.0,2.5],[1.5,2.0,2.5]],
#    [[0.0,0.5,1.0],[1.5,2.0,2.5],[1.5,2.0,2.5],[0.0,0.5,1.0]],
#])
#key = tf.constant([
#    [[0.0,0.5,1.0],[0.0,0.5,1.0],[1.5,2.0,2.5],[1.5,2.0,2.5]],
#    [[1.5,2.0,2.5],[0.0,0.5,1.0],[0.0,0.5,1.0],[1.5,2.0,2.5]],
#])
#query_mask = tf.constant([
#    [True,True],
#    [True,False],
#])
#value_mask = tf.constant([
#    [True,True,True,True],
#    [True,True,False,True],
#])
#
##################################################
#[vector,scores] = attention(
#    [query,value,key],
#    return_attention_scores=True)
#
##print('query=',query)
##print('value=',value)
##print('query_mask=',query_mask)
##print('value_mask=',value_mask)
#print('vector=',vector)
#print('scores=',scores)
#
##################################################
#[vector,scores] = attention(
#    [query,value,key],
#    mask=[None,value_mask],#[query_mask,value_mask],
#    return_attention_scores=True)
#
##print('query=',query)
##print('value=',value)
##print('query_mask=',query_mask)
##print('value_mask=',value_mask)
#print('vector=',vector)
#print('scores=',scores)
#

#################################################
print('#################################################')
query = tf.Variable([
            [[1,0,0],[0,1,0]],
            [[1,0,0],[0,1,0]],
],dtype=tf.float32)
value = tf.Variable([
            [[1,0,0],[0,1,0],[0,0,1],[0,0,0]],
            [[1,0,0],[0,1,0],[0,0,1],[0,0,0]],
],dtype=tf.float32)
query_mask = tf.constant([
            [True,False],
            [False,True],
])
value_mask = tf.constant([
            [False,False,True,True],
            [False,True,True,False],
])
with tf.GradientTape() as tape:
    [vector,scores] = attention(
        [query,value],
        mask=[query_mask,value_mask],
        return_attention_scores=True)
print('vector=',vector)
print('scores=',scores)
grads = tape.gradient(vector,[query,value])
print('dQuery=',grads[0])
print('dValue=',grads[1])

