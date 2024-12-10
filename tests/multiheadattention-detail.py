import tensorflow as tf
from functools import reduce
from operator import mul

num_heads = 8
key_dim = 4
full_query_shape = [2, 6, 16]
full_value_shape = [2, 7, 16]


query_dense_equation  =  'abc,cde->abde'
key_dense_equation    =  'abc,cde->abde'
value_dense_equation  =  'abc,cde->abde'
dot_product_equation  = 'aecd,abcd->acbe'
combine_equation      = 'acbe,aecd->abcd'
output_dense_equation = 'abcd,cde->abe'

query_output_shape  = [None, 8, 4]
key_output_shape    = [None, 8, 4]
value_output_shape  = [None, 8, 4]
output_output_shape = [None, 16]

query_dense = tf.keras.layers.EinsumDense(
    query_dense_equation,
    query_output_shape,
    kernel_initializer="ones",
    bias_initializer="zeros",
)
key_dense = tf.keras.layers.EinsumDense(
    key_dense_equation,
    key_output_shape,
    kernel_initializer="ones",
    bias_initializer="zeros",
)
value_dense = tf.keras.layers.EinsumDense(
    value_dense_equation,
    value_output_shape,
    kernel_initializer="ones",
    bias_initializer="zeros",
)
output_dense = tf.keras.layers.EinsumDense(
    output_dense_equation,
    output_output_shape,
    kernel_initializer="ones",
    bias_initializer="zeros",
)

query = tf.Variable(tf.ones(full_query_shape))  # (batch_size, context_len, d_model)
value = tf.Variable(tf.ones(full_value_shape))  # (batch_size, context_len, d_model)
key = tf.Variable(tf.ones(full_value_shape))  # (batch_size, context_len, d_model)
alp = tf.reshape(tf.range(0, reduce(mul, full_query_shape), dtype=tf.float32),full_query_shape)


with tf.GradientTape() as tape:
    query_ = query_dense(query)
    key_ = key_dense(key)
    value_ = value_dense(value)

    query_ = tf.multiply(query_, 1.0 / tf.math.sqrt(float(key_dim)))

    attention_scores = tf.einsum(dot_product_equation, key_, query_)
    attention_scores = tf.nn.softmax(attention_scores)

    #attention_scores_dropout = self._dropout_layer(attention_scores, training=True)
    attention_scores_dropout = attention_scores

    attention_output = tf.einsum(combine_equation, attention_scores_dropout, value_)

    attention_output = output_dense(attention_output)

    results =  attention_output * alp

#print('attention_output: ',attention_output)
#print('attention_scores: ',attention_scores)

grads = tape.gradient(results, [query,value,key])
#grads = tape.gradient(attention_output, [query,value,key])

d_query, d_value, d_key = grads

#print('query:', query)
print('d_query:', d_query)
