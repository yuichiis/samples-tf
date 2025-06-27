import tensorflow as tf
from functools import reduce
from operator import mul

num_heads = 8
key_dim = 4
batches = 2
Tq = [6]
Tv = [7]
attention_axes = (1,)
#Tq = [3,6]
#Tv = [3,7]
#attention_axes = (1,2,)
detail = 5
query_mask = tf.constant([
    [True,True,True,False,False,False],
    [True,True,True,True,True,False],
])
value_mask = tf.constant([
    [True,True,True,True,False,False,False],
    [True,True,True,True,True,True,False],
])

full_query_shape = [batches]+Tq+[detail]
full_value_shape = [batches]+Tv+[detail]

#shape = [2,2,2,2,2]
#a = tf.reshape(tf.range(0, reduce(mul, shape), dtype=tf.float32),shape)
#b = tf.reshape(tf.range(0, reduce(mul, shape), dtype=tf.float32),shape)
#print(tf.einsum('afgde,abcde->adbcfg',a,b))

#shapeA = [2,2]
#shapeB = [2,3]
#a = tf.reshape(tf.range(1, 1+reduce(mul, shapeA), dtype=tf.float32),shapeA)
#b = tf.reshape(tf.range(1, 1+reduce(mul, shapeB), dtype=tf.float32),shapeB)
#print(tf.einsum('aa,ab->ab',a,b))

#shapeA = [2,3]
#shapeB = [3]
#a = tf.reshape(tf.range(1, 1+reduce(mul, shapeA), dtype=tf.float32),shapeA)
#b = tf.reshape(tf.range(1, 1+reduce(mul, shapeB), dtype=tf.float32),shapeB)
#print(tf.einsum('ab,b->ab',a,b))

#shapeA = [2,3]
#shapeB = [1,3]
#a = tf.reshape(tf.range(1, 1+reduce(mul, shapeA), dtype=tf.float32),shapeA)
#b = tf.reshape(tf.range(1, 1+reduce(mul, shapeB), dtype=tf.float32),shapeB)
#print(tf.einsum('...a,...a->...a',a,b))

#shapeA = [1,3]
#shapeB = [2,3]
#a = tf.reshape(tf.range(1, 1+reduce(mul, shapeA), dtype=tf.float32),shapeA)
#b = tf.reshape(tf.range(1, 1+reduce(mul, shapeB), dtype=tf.float32),shapeB)
#print(tf.einsum('...a,...a->...a',a,b))
#exit()

query_dense_equation  = 'abc,cde->abde'     # (B.Tq.Fq),(Fq.H.Dk)->(B.Tq.H.Dk)
key_dense_equation    = 'abc,cde->abde'     # (B.Tv.Fk),(Fk.H.Dk)->(B.Tv.H.Dk)
value_dense_equation  = 'abc,cde->abde'     # (B.Tv.Fv),(Fv.H.Dv)->(B.Tv.H.Dv)
dot_product_equation  = 'aecd,abcd->acbe'   # key(B.Tv.H.Dk),query(B.Tq.H.Dk)->scores(B.H.Tq.Tv) #(key,query->prod_attn)
combine_equation      = 'acbe,aecd->abcd'   # scores(B.H.Tq.Tv),value(B.Tv.H.Dv)->output(B.Tq.H.Dv)
output_dense_equation = 'abcd,cde->abe'     # (B.Tq.H.Dv),(H.Dv.Fq)->(B.Tq.Fq)

#shape_a = (2,2)
#shape_b = (2,2)
#a = tf.reshape(tf.range(1, 1+reduce(mul, shape_a), dtype=tf.float32),shape_a)
#b = tf.reshape(tf.range(1, 1+reduce(mul, shape_b), dtype=tf.float32),shape_b)
#print(a.shape)
#print(b.shape)
#print(tf.einsum('ab,ab->ab',a,b))
#exit()

#query_dense_equation  = 'abcd,def->abcef'
#key_dense_equation    = 'abcd,def->abcef'
#value_dense_equation  = 'abcd,def->abcef'
#dot_product_equation  = 'afgde,abcde->adbcfg'
#combine_equation      = 'adbcfg,afgde->abcde'
#output_dense_equation = 'abcde,def->abcf'
#attention_axes        = (1, 2)
null_axes = []
for _ in range(len(Tq)):
    null_axes.append(None)

query_output_shape  = null_axes+[num_heads, key_dim]
key_output_shape    = null_axes+[num_heads, key_dim]
value_output_shape  = null_axes+[num_heads, key_dim]
output_output_shape = null_axes+[detail]
attn_scores_rank = 1+1+len(Tq)+len(Tv)  # batchs+head+Tq+Tv
norm_axes = tuple(
    range(
        attn_scores_rank - len(attention_axes), attn_scores_rank
    )
)
print('attn_scores_rank',attn_scores_rank)
print('attention_axes',attention_axes)
print('norm_axes',norm_axes)


#full_query_shape = [batches, T0q, T1q, detail]  # [batches, T0q, T1q, detail] # attention dims [T0q, T1q]
#full_value_shape = [batches, T0v, T1v, detail]  # [batches, T0v, T1v, detail] # attention dims [T0v, T1v]

#query_dense_equation  =  'abcd,def->abcef'
#key_dense_equation    =  'abcd,def->abcef'
#value_dense_equation  =  'abcd,def->abcef'
#dot_product_equation  = 'afgde,abcde->adbcfg'
#combine_equation      = 'adbcfg,afgde->abcde'
#output_dense_equation = 'abcde,def->abcf'

#query_output_shape  = [None, None, num_heads, key_dim]
#key_output_shape    = [None, None, num_heads, key_dim]
#value_output_shape  = [None, None, num_heads, key_dim]
#output_output_shape = [None, None, detail]

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
def compute_causal_mask(query,value) :
    #n_attn_axes = len(attention_axes)
    q_seq_length = tf.shape(query)[1]
    v_seq_length = q_seq_length if value is None else tf.shape(value)[1]
    ones_mask = tf.ones((1, q_seq_length, v_seq_length), dtype="int32")
    row_index = tf.cumsum(ones_mask, axis=-2)
    col_index = tf.cumsum(ones_mask, axis=-1)
    return tf.greater_equal(row_index, col_index)
    

#softmax = tf.keras.layers.Softmax(axis=norm_axes)

alp_q = tf.reshape(tf.range(0, reduce(mul, full_query_shape), dtype=tf.float32),full_query_shape)
alp_v = tf.reshape(tf.range(0, reduce(mul, full_value_shape), dtype=tf.float32),full_value_shape)
query = tf.Variable((alp_q+1)/reduce(mul, full_query_shape))  # (batch_size, context_len, d_model)
value = tf.Variable((alp_v+1)/reduce(mul, full_value_shape))  # (batch_size, context_len, d_model)
#key   = tf.Variable((alp_v+1)/reduce(mul, full_value_shape))  # (batch_size, context_len, d_model)
#query = tf.random.normal(full_query_shape,dtype=tf.float32)
#value = tf.random.normal(full_value_shape,dtype=tf.float32)
#key = tf.random.normal(full_value_shape,dtype=tf.float32)

print('query:', query.shape)
print('value:', value.shape)

def custom_softmax_log_softmax(x):
    return tf.exp(tf.nn.log_softmax(x, axis=-1))

def custom_softmax_clip(x):
    submax = x - tf.reduce_max(x, axis=-1, keepdims=True)
    exp_x = tf.exp(tf.clip_by_value(submax, -100.0, 100.0)) # clip values
    sumexp = tf.reduce_sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sumexp

def custom_softmax(x) :
    submax = x - tf.reduce_max(x, axis=-1, keepdims=True)
    exp_x = tf.exp(submax)
    sumexp = tf.reduce_sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sumexp

def custom_softmax2(x) :
    submax = x - tf.reduce_max(x, axis=-1, keepdims=True)
    exp_x = tf.exp(submax)
    sumexp = tf.reduce_sum(exp_x, axis=-1, keepdims=True)
    return exp_x (1/sumexp)

with tf.GradientTape() as tape:
    key = value
    query_ = query_dense(query)
    key_ = key_dense(key)
    value_ = value_dense(value)

    scaled_query = tf.multiply(query_, 1.0 / tf.math.sqrt(float(key_dim)))

    attention_scores = tf.einsum(dot_product_equation, key_, scaled_query)
    print('attention_scores',attention_scores.shape)


    print('query_mask',query_mask.shape)
    print('value_mask',value_mask.shape)
    auto_mask = tf.expand_dims(query_mask, -1)  # shape is [B, T, 1]
    print('reshaped_query_mask',tf.expand_dims(query_mask, -1).shape)
    auto_mask = auto_mask & tf.expand_dims(value_mask, -2)  # shape is [B, T, 1]
    #auto_mask = tf.expand_dims(value_mask, -2)  # shape is [B, T, 1]
    print('reshaped_value_mask',tf.expand_dims(value_mask, -2).shape)
    auto_mask = auto_mask & compute_causal_mask(query,value)
    attention_mask = auto_mask
    #print('auto_mask',auto_mask)
    print('auto_mask',auto_mask.shape)
    
    mask_expansion_axis = -len(attention_axes) * 2 - 1
    print('mask_expansion_axis',mask_expansion_axis)
    for _ in range(
        len(attention_scores.shape) - len(attention_mask.shape)
    ):
        attention_mask = tf.expand_dims(
            attention_mask, axis=mask_expansion_axis
        )
    #print('reshape_attention_mask',attention_mask)
    print('reshaped_attention_mask',attention_mask.shape)
    #print('norm_axes',norm_axes)

    org_attention_mask = attention_mask
    attention_mask = -1e9*tf.cast(tf.math.logical_not(attention_mask),tf.float32)
    #print('attention_mask',attention_mask)
    masked_attention_scores = attention_scores + attention_mask
    #print('masked_attention_scores',masked_attention_scores)
    #softmax_attention_scores = tf.nn.softmax(attention_scores)
    attention_scores_flat = tf.reshape(masked_attention_scores, tuple([batches,num_heads]+Tq+[reduce(mul,Tv)]))
    #print('attention_scores_flat',attention_scores_flat.shape)

    softmax_attention_scores = tf.nn.softmax(attention_scores_flat)
    #softmax_attention_scores = custom_softmax(attention_scores_flat)

    #print(tf.math.reduce_sum(
    #print(    softmax_attention_scores-softmax_attention_scores_custom)
    #axis=-1))

    #print(tf.math.reduce_sum(softmax_attention_scores,axis=1))
    #softmax_attention_scores = softmax(attention_scores,mask=attention_mask)
    #softmax_attention_scores = softmax(attention_scores)#,mask=attention_mask)
    softmax_attention_scores = tf.reshape(softmax_attention_scores,tuple([batches,num_heads]+Tq+Tv))
    #softmax_attention_scores = softmax_attention_scores*tf.cast(org_attention_mask,dtype=tf.float32)

    attention_scores_dropout = softmax_attention_scores

    t_attention_output = tf.einsum(combine_equation, attention_scores_dropout, value_)

    attention_output = output_dense(t_attention_output)

    results =  attention_output * alp_q

grads = tape.gradient(results, [
    query,value,key,
    query_,value_,key_,
    scaled_query,
    attention_scores,
    masked_attention_scores,
    softmax_attention_scores,
    t_attention_output,
    attention_output,
])
#grads = tape.gradient(attention_output, [query,value,key])

[
    d_query, d_value, d_key,
    d_query_,d_value_,d_key_,
    d_scaled_query,
    d_attention_scores,
    d_masked_attention_scores,
    d_softmax_attention_scores,
    d_t_attention_output,
    d_attention_output,
] = grads

print('dot_product',dot_product_equation)

#print(query_dense.kernel)
#print(query_dense.bias)
#print(query_dense.equation)
#print(query_dense.bias_axes)

#print('query:', query)
#print('query_:', query_)
#print('query_:', tf.math.reduce_sum(tf.math.reduce_sum(query_,axis=-1),axis=-1))
#print('value:', value)
#print('value_:', value_)
#print('value_:', tf.math.reduce_sum(tf.math.reduce_sum(value_,axis=-1),axis=-1))
#print('key:', key)
#print('key_:', key_)
#print('scaled_query:', scaled_query)
#print('scaled_query:', tf.math.reduce_sum(tf.math.reduce_sum(scaled_query,axis=-1),axis=-1))
#print('attention_output:',attention_output)
#print('attention_output:',tf.math.reduce_sum(attention_output,axis=-1))
#print('attention_scores:',attention_scores)
#print('attention_scores:',tf.math.reduce_sum(attention_scores,axis=1))
#print('masked_attention_scores:',masked_attention_scores)
#print('masked_attention_scores:',tf.math.reduce_sum(masked_attention_scores,axis=1))
#print('attention_scores0:',attention_scores[:,0,:,:])
#print('attention_scores-1:',attention_scores[:,-1,:,:])
#print('softmax_attention_scores:',softmax_attention_scores)
#print('softmax_attention_scores:',tf.math.reduce_sum(softmax_attention_scores,axis=1))
#print('softmax_attention_scores:',softmax_attention_scores[0,0,0,:,:,:])
#print('softmax_attention_scores0:',softmax_attention_scores[:,0,:,:])
#print('softmax_attention_scores-1:',softmax_attention_scores[:,-1,:,:])
#print('softmax_attention_scores.shape:',softmax_attention_scores.shape)
#print('t_attention_output:', t_attention_output)
#print('t_attention_output:', tf.math.reduce_sum(tf.math.reduce_sum(t_attention_output,axis=-1),axis=-1))
#print('d_output:', d_attention_output)
#print('d_t_attention_output:', d_t_attention_output)
#print('d_t_attention_output:', tf.math.reduce_sum(tf.math.reduce_sum(d_t_attention_output,axis=-1),axis=-1))
#print('d_masked_attention_scores:',tf.math.reduce_sum(d_masked_attention_scores,axis=1))
#print('d_softmax_attention_scores:', d_softmax_attention_scores)
#print('d_softmax_attention_scores:', tf.math.reduce_sum(d_softmax_attention_scores,axis=1))
#print('softmax_attention_scores:',softmax_attention_scores)
#print('dot_product_equation_forward:', dot_product_equation)
#print('d_attention_scores:', d_attention_scores)
#print('d_attention_scores:',tf.math.reduce_sum(d_attention_scores,axis=1))
#print('query_:', query_)
#print('key_:', key_)
#print('key_:', tf.math.reduce_sum(tf.math.reduce_sum(key_,axis=-1),axis=-1))
#print('d_key_:', d_key_)
#print('d_key_:', tf.math.reduce_sum(tf.math.reduce_sum(d_key_,axis=-1),axis=-1))
#print('d_scaled_query:', d_scaled_query)
#print('d_scaled_query:', tf.math.reduce_sum(tf.math.reduce_sum(d_scaled_query,axis=-1),axis=-1))

#print('d_query_:', d_query_)
#print('d_query:', d_query)
#print('d_value_:', d_value_)
#print('d_value_:', tf.math.reduce_sum(tf.math.reduce_sum(d_value_,axis=-1),axis=-1))
print('d_value:', d_value)
