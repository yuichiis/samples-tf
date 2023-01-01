import tensorflow as tf
import numpy as np

def func0(a,b):
    print('func0(',a,b,')++')
    x = a+b
    y = tf.keras.layers.Dense(3,)(x)
    print('func0--')
    return y

def func1(b,c):
    print('func1(',b,c,')++')
    c0 = c+c
    print('  c0:',c0)
    x = func0(c0,b)
    print('  x:',x)
    y = x*c
    print('  y:',y)
    print('func1--')
    return y


#a = tf.Variable([[2.],[3.]])
#b = tf.Variable([[2.],[4.]])
a = tf.Variable([[2.],[3.],[3.]])
b = tf.Variable([[2.],[4.],[3.]])

l = tf.keras.layers.Dense(3,kernel_initializer='ones')
with tf.GradientTape() as tape:
    x = a+b
    y = l(x)

print('y=func1',y)

#print(l.weights)
grads = tape.gradient(y,[a,b]+l.weights)
print('grads(func1)',grads)

