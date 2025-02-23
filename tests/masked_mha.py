import tensorflow as tf
from functools import reduce
from operator import mul

num_heads = 8
key_dim = 4

input_dim = 10
output_dim = 16
query_length = 6
value_length = 7

query_seq = [
    [ 1, 2, 3, 4, 0, 0],
    [ 1, 2, 3, 4, 5, 0],
    [ 1, 2, 3, 4, 5, 6],
    [ 1, 2, 3, 4, 5, 6],
]
value_seq = [
    [ 1, 2, 3, 4, 0, 0, 0],
    [ 1, 2, 3, 4, 5, 0, 0],
    [ 1, 2, 3, 4, 5, 6, 0],
    [ 1, 2, 3, 4, 5, 6, 7],
]
#full_query_shape = [len(query_seq), query_length, output_dim]
#full_value_shape = [len(value_seq), value_length, output_dim]
full_query_shape = [2, 3, 6, output_dim]
full_value_shape = [2, 3, 7, output_dim]

class MyCustomLayer(tf.keras.layers.Layer):
    def __init__(self, layername=None):
        self.layername = layername
        super(MyCustomLayer, self).__init__()
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        #print(inputs)
        if mask is not None:
            print("Input mask (retrieved in "+self.layername+".compute_mask):", mask)
            mask = tf.math.logical_not(mask)
        else:
            print("No mask (in "+self.layername+".compute_mask)")
        return mask

    def call(self, inputs, mask=None):

        if mask is not None:
            print("Original mask (retrieved in "+self.layername+".call):", mask)
            inputs += 1
        else:
            print("No mask (in "+self.layername+".call)")
        return inputs


emb_q = tf.keras.layers.Embedding(
    input_dim, output_dim, mask_zero=False, input_length=query_length
)
emb_v = tf.keras.layers.Embedding(
    input_dim, output_dim, mask_zero=False, input_length=value_length
)
customFirst = MyCustomLayer(layername='First')
customSecond = MyCustomLayer(layername='Second')
mha = tf.keras.layers.MultiHeadAttention(
    num_heads,
    key_dim,
    kernel_initializer="ones",
    bias_initializer="zeros",
)

query = tf.Variable(tf.ones(full_query_shape))  # (batch_size, context_len, d_model)
value = tf.Variable(tf.ones(full_value_shape))
#query = tf.random.normal(full_query_shape,dtype=tf.float32)
#value = tf.random.normal(full_value_shape,dtype=tf.float32)
#query_seq = tf.constant(query_seq,dtype=tf.int32)
#value_seq = tf.constant(value_seq,dtype=tf.int32)
#query_mask = tf.cast(query_seq,tf.bool)
#value_mask = tf.cast(value_seq,tf.bool)
query_seq = tf.Variable(query_seq)
value_seq = tf.Variable(value_seq)
#query_mask = tf.Variable(query_mask)
#value_mask = tf.Variable(value_mask)

alp = tf.reshape(tf.range(0, reduce(mul, full_query_shape), dtype=tf.float32),full_query_shape)

with tf.GradientTape() as tape:
    #query = emb_q(query_seq)
    #value = emb_v(value_seq)
    query = customFirst(query)
    value = customFirst(value)
    outputs, scores = mha( # output: B,T,E   scores: ?,?,?
        query, value,
        return_attention_scores=True,
        #use_causal_mask=True,
    )
    x = customSecond(outputs)
    results =  x * alp

print("outputs: ", outputs)
print("outputs: ", outputs.shape)
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
