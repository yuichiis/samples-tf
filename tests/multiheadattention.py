import tensorflow as tf
from functools import reduce
from operator import mul

num_heads = 8
key_dim = 4
full_query_shape = [2, 6, 16]
full_value_shape = [2, 7, 16]


mha = tf.keras.layers.MultiHeadAttention(
    num_heads,
    key_dim,
    #kernel_initializer="ones",
    #bias_initializer="zeros",
        
)

#query = tf.Variable(tf.ones(full_query_shape))  # (batch_size, context_len, d_model)
#value = tf.Variable(tf.ones(full_value_shape))
query = tf.random.normal(full_query_shape,dtype=tf.float32)
value = tf.random.normal(full_value_shape,dtype=tf.float32)

alp = tf.reshape(tf.range(0, reduce(mul, full_query_shape), dtype=tf.float32),full_query_shape)

with tf.GradientTape() as tape:
    outputs, scores = mha( # output: B,T,E   scores: ?,?,?
        query, value,
        return_attention_scores=True,
        #use_causal_mask=True,
    )
    results =  outputs * alp

print("outputs: {%.4f}" % outputs.numpy())
#print(outputs.shape)
#print('scores:',scores)
#print(scores.shape)

#grads = tape.gradient(outputs,[query,value])
grads = tape.gradient(results,[
    query,
    value,
])
[
    d_query,
    d_value,
] = grads
#print('query:', query)
#print('d_query:',d_query)
#print(d_query.shape)
#print('d_value:',d_value)
#print(d_value.shape)
