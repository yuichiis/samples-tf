import tensorflow as tf
from functools import reduce
from operator import mul

num_heads = 8
key_dim = 4
batches = 2
detail = 5
Tq = [6]
Tv = [7]
#Tq = [3,6]
#Tv = [3,7]

mask_q = tf.constant([
    [True,True,True,False,False,False],
    [True,True,True,True,True,False],
])
mask_v = tf.constant([
    [True,True,True,True,False,False,False],
    [True,True,True,True,True,True,False],
])
#mask_k = tf.constant([
#    [True,True,True,True,False,False,False],
#    [True,True,True,True,True,True,False],
#])
#mask_q = tf.constant([
#   [[True,True,True,False,False,False],
#    [True,True,True,False,False,False],
#    [True,True,True,False,False,False]],
#   [[True,True,True,True,True,False],
#    [True,True,True,True,True,False],
#    [True,True,True,True,True,False]],
#])
#mask_v = tf.constant([
#   [[True,True,True,True,False,False,False],
#    [True,True,True,True,False,False,False],
#    [True,True,True,True,False,False,False]],
#   [[True,True,True,True,True,True,False],
#    [True,True,True,True,True,True,False],
#    [True,True,True,True,True,True,False]],
#])



#full_query_shape = [2, 3, 6, 6]  # [batches, T0q, T1q, detail] # attention dims [2, 6]
#full_value_shape = [2, 3, 7, 6]  # [batches, T0v, T1v, detail] # attention dims [3, 7]
#full_query_shape = [2, 6, 16]  # [batches, T0q, T1q, detail] # attention dims [2, 6]
#full_value_shape = [2, 7, 16]  # [batches, T0v, T1v, detail] # attention dims [3, 7]
#full_query_shape = [2, 3, 4]  # [batches, T0q, T1q, detail] # attention dims [2, 6]
#full_value_shape = [2, 5, 4]  # [batches, T0v, T1v, detail] # attention dims [3, 7]
#full_query_shape = [2, 3, 16]  # [batches, T0q, T1q, detail] # attention dims [2, 6]
#full_value_shape = [2, 5, 16]  # [batches, T0v, T1v, detail] # attention dims [3, 7]
full_query_shape = [batches]+Tq+[detail]
full_value_shape = [batches]+Tv+[detail]


mha = tf.keras.layers.MultiHeadAttention(
    num_heads,
    key_dim,
    kernel_initializer="ones",
    bias_initializer="zeros",
    #attention_axes=(1,2)
)
class ApplyMask(tf.keras.layers.Layer):
    def __init__(self, layername=None):
        self.layername = layername
        super(ApplyMask, self).__init__()

    def compute_mask(self, inputs, mask=None):
        #if mask is not None:
        #    print("Input ", inputs.shape, ", mask (retrieved in "+self.layername+".compute_mask):", mask)
        #else:
        #    print("No mask (in "+self.layername+".compute_mask)")
        return mask

    def call(self, inputs, mask=None):
        #if mask is not None:
        #    print("Input ", inputs.shape, ", Original mask (retrieved in "+self.layername+".call):", mask)
        #    #inputs = inputs*tf.expand_dims(tf.cast(mask,tf.float32),axis=-1)
        #else:
        #    print("No mask (in "+self.layername+".call)")
        return inputs
apply_mask_q = ApplyMask('ApplyMaskQ')
apply_mask_v = ApplyMask('ApplyMaskV')
#apply_mask_k = ApplyMask('ApplyMaskK')

alp_q = tf.reshape(tf.range(0, reduce(mul, full_query_shape), dtype=tf.float32),full_query_shape)
alp_v = tf.reshape(tf.range(0, reduce(mul, full_value_shape), dtype=tf.float32),full_value_shape)

query = tf.Variable((alp_q+1)/reduce(mul, full_query_shape))  # (batch_size, context_len, d_model)
value = tf.Variable((alp_v+1)/reduce(mul, full_value_shape))  # (batch_size, context_len, d_model)
#key = tf.Variable((alp_v+1)/reduce(mul, full_value_shape))  # (batch_size, context_len, d_model)

#print(value)
#query_emb = tf.keras.layers.Embedding(
#    16,
#    4,
#    mask_zero=True,
#    embeddings_initializer="ones",
#)
#value_emb = tf.keras.layers.Embedding(
#    16,
#    4,
#    mask_zero=True,
#    embeddings_initializer="ones",
#)
#query_seq = tf.Variable(tf.constant([
#    [  1,  1,  0],
#    [  1,  0,  0],
#],tf.int32))
##query_seq = tf.Variable(tf.constant([
##    [  1,  1,  1],
##    [  1,  1,  1],
##],tf.int32))
#value_seq = tf.Variable(tf.constant([
#    [  1,  1,  0,  0,  0],
#    [  1,  1,  1,  1,  0],
#],tf.int32))
##value_seq = tf.Variable(tf.constant([
##    [  1,  1,  1,  1,  1],
##    [  1,  1,  1,  1,  1],
##],tf.int32))

#queryMask = tf.constant([ # (2,3)
#    [True,True, False],
#    [True,False,False],
#],dtype=tf.bool)
#valueMask = tf.constant([ # (2,5)
#    [True,True,False,False,False],
#    [True,True,True, True, False],
#],dtype=tf.bool)


print('query:', query.shape)
print('value:', value.shape)
#print('key:', key.shape)
print('mask_q:', mask_q.shape)
print('mask_v:', mask_v.shape)
#print('mask_k:', mask_k.shape)

with tf.GradientTape() as tape:
    #query = query_emb(query_seq)
    #value = value_emb(value_seq)
    query = apply_mask_q(query,mask=mask_q)
    value = apply_mask_q(value,mask=mask_v)
    #key = apply_mask_q(key,mask=mask_v)

    outputs, scores = mha( # output: B,T,E   scores: ?,?,?
        query, value,
        #key,
        return_attention_scores=True,
        training=True,
        use_causal_mask=True,
        #mask=None,
        #query_mask=queryMask,
        #value_mask=valueMask,
    )
    results =  outputs * alp_q

#print("outputs: {%.4f}" % outputs.numpy())
#print(outputs.shape)
#print('scores:',scores)
#print(scores.shape)

#grads = tape.gradient(outputs,[query,value])
grads = tape.gradient(results,[
    query,
    value,
    #key,
])
[
    d_query,
    d_value,
    #d_key,
] = grads
#print('outputs:',outputs)
#print('attention_scores:',scores)
#print('attention_scores:',tf.math.reduce_sum(scores,axis=1))
#print('scores0:',scores[0,0,0,:,:,:])
#print('scores-1:',scores[-1,-1,-1,:,:,:])
#print('scores0:',scores[:,0,:,:])
#print('scores-1:',scores[:,-1,:,:])
#print('scores.shape:',scores.shape)

#print('query:', query)
#print('d_query:',d_query)
#print(d_query.shape)
print('d_value:',d_value)
#print(d_value.shape)
#print('d_key:',d_key)
#print(d_key.shape)



#print('query:                ',query.shape)
#print('value:                ',value.shape)
#print('num_heads:            ',mha._num_heads)
#print('key_dim:              ',mha._key_dim)
#print('value_dim:            ',mha._value_dim)
#print('query_dense:          ',mha._query_dense.equation,'  (),',mha._query_dense.kernel.shape,'=>',tuple(mha._query_dense.full_output_shape))
#print('key_dense:            ',mha._key_dense.equation,'  (),',mha._key_dense.kernel.shape,'=>',tuple(mha._key_dense.full_output_shape))
#print('value_dense:          ',mha._value_dense.equation,'  (),',mha._value_dense.kernel.shape,'=>',tuple(mha._value_dense.full_output_shape))
#print('dot_product_equation: ',mha._dot_product_equation)
#print('combine_equation:     ',mha._combine_equation)
#print('output_dense:         ',mha._output_dense.equation,'  (),',mha._output_dense.kernel.shape,'=>',tuple(mha._output_dense.full_output_shape))
##print('output_dense:         ',mha._output_dense.equation,'  (),',mha._output_dense.kernel.shape,'=>',tuple(mha._output_dense.compute_output_shape(query.shape)))
#print('attention_axes:       ',mha._attention_axes)
#print('output_shape:         ',mha._output_shape)
