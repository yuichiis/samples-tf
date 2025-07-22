import tensorflow as tf

emb = tf.keras.layers.Embedding(
    5, # inputDim
    4, # outputDim
    mask_zero=True,
    embeddings_initializer="ones",
)
dense = tf.keras.layers.Dense(
    5,
    kernel_initializer='ones',
)
inputs = tf.Variable([
    [1,2,3,4,0,0,0,0,],
    [1,2,3,4,1,2,3,4,],
],dtype=tf.int32)

with tf.GradientTape() as tape:
    x = emb(inputs)
    outputs = dense(x)

#print(emb.trainable_variables)
#print(dense.trainable_variables)
grads = tape.gradient(outputs, [inputs,x]+emb.trainable_variables+dense.trainable_variables)
dInputs, dx, demb, ddensek, ddenseb = grads
print(dx)
print(ddensek)
#sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
#sgd.apply_gradients(zip([dkernel], emb.trainable_variables))
#print(emb.trainable_variables)

#print(inputs.shape, '->', outputs.shape)
#print(outputs)
#print(dInputs)
