import tensorflow as tf
from functools import reduce
from operator import mul


equation = 'ab,bc->ac'
bias_axes = 'c'
full_input_shape = [4,3]
output_shape = [2]
full_output_shape = [4] + output_shape

# class EinsumDense:
#    def __init__(
#        self,
#        equation,
#        output_shape,
#        activation=None,
#        bias_axes=None,
#        kernel_initializer="glorot_uniform",
#        bias_initializer="zeros",
#        kernel_regularizer=None,
#        bias_regularizer=None,
#        kernel_constraint=None,
#        bias_constraint=None,
#        lora_rank=None,
#        **kwargs,
#    ):

dense = tf.keras.layers.EinsumDense(
    equation,
    output_shape,
    kernel_initializer="ones",
    bias_initializer="zeros",
        
)

#inputs = tf.Variable(tf.ones(full_input_shape))  # (batch_size, d_model)
inputs = tf.Variable([
    [0.1, 0.2, 0.5],
    [0.1, 0.2, 0.5],
    [0.1, 0.2, 0.5],
    [0.1, 0.2, 0.5],
],dtype=tf.float32)
print('inputs:',inputs)

salt = tf.reshape(tf.range(0, reduce(mul, full_output_shape), dtype=tf.float32), full_output_shape)
#print('salt:',salt)

with tf.GradientTape() as tape:
    outputs = dense( # 
        inputs
    )
    results =  outputs * salt

print('outputs:',outputs)
#print(outputs.shape)

grads = tape.gradient(results, [inputs, outputs])
d_inputs, d_outputs = grads
print('d_inputs:',d_inputs)
print(d_inputs.shape)
#print('d_outputs:',d_outputs)
#print(d_outputs.shape)
