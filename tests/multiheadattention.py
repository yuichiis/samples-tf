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
    kernel_initializer="ones",
    bias_initializer="zeros",
        
)

query = tf.Variable(tf.ones(full_query_shape))  # (batch_size, context_len, d_model)
value = tf.Variable(tf.ones(full_value_shape))
#key = tf.Variable(tf.ones([8,32,128]))
#query = tf.Variable(
#    1.0 + 0.0625*tf.reshape(tf.range(0, reduce(mul, full_query_shape), dtype=tf.float32),full_query_shape)
#)
#value = tf.Variable(
#    1.0 + 0.04*tf.reshape(tf.range(0, reduce(mul, full_query_shape), dtype=tf.float32),full_query_shape)
#)
#print('query:',query)
#print('value:',value)


#    def call(
#        self,
#        query,
#        value,
#        key=None,
#        query_mask=None,
#        value_mask=None,
#        key_mask=None,
#        attention_mask=None,
#        return_attention_scores=False,
#        training=None,
#        use_causal_mask=False,
#    ):

alp = tf.reshape(tf.range(0, reduce(mul, full_query_shape), dtype=tf.float32),full_query_shape)

with tf.GradientTape() as tape:
    outputs, scores = mha( # output: B,T,E   scores: ?,?,?
        query, value,
        return_attention_scores=True,
        #use_causal_mask=True,
    )
    results =  outputs * alp

#print('outputs:',outputs)
#print(outputs.shape)
#print('scores:',scores)
#print(scores.shape)

#grads = tape.gradient(outputs,[query,value])
grads = tape.gradient(results,[query,value])
d_query = grads[0]
d_value = grads[1]
#print('query:', query)
print('d_query:',d_query)
#print(d_query.shape)
#print('d_value:',d_value)
#print(d_value.shape)
