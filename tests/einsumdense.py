import tensorflow as tf
from functools import reduce
from operator import mul

equation = 'abc,cde->abde'
full_input_shape = [2, 6, 16]
virtual_output_shape  = [None, 8, 4]
full_output_shape  = [2, 6, 8, 4]

dense = tf.keras.layers.EinsumDense(
    equation,
    virtual_output_shape,
)

inputs = tf.Variable(tf.ones(full_input_shape))
print('inputs:', inputs.shape)

salt = tf.reshape(tf.range(0, reduce(mul, full_output_shape), dtype=tf.float32), full_output_shape)
#print('salt:',salt)

with tf.GradientTape() as tape:
    outputs = dense( # 
        inputs
    )
    results =  outputs * salt

print('kernel:', dense.equation)
#print('kernel:', dense.kernel)
print('kernel:', dense.kernel.shape)
print('outputs:',outputs.shape)

grads = tape.gradient(results, [inputs, outputs])
d_inputs, d_outputs = grads
#print('d_inputs:',d_inputs)
#print(d_inputs.shape)
#print('d_outputs:',d_outputs)
#print(d_outputs.shape)
#a = tf.reshape(outputs,[4,2])
#print('reduce_sum(outputs)',tf.math.reduce_sum(a,axis=-1,keepdims=True))
#a = tf.reshape(outputs,[4,3*2])
#print('reduce_sum(outputs)',tf.math.reduce_sum(a,axis=-1,keepdims=True))
print(dense.kernel_initializer)
