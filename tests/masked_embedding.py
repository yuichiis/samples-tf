import tensorflow as tf

emb = tf.keras.layers.Embedding(
    5, # inputDim
    4, # outputDim
    #mask_zero=True,
    embeddings_initializer="ones",
)

inputs = tf.Variable([
    [1,2,3,4,0,0,0,0,],
    [1,2,3,4,1,2,3,4,],
],dtype=tf.int32)

with tf.GradientTape() as tape:
    outputs = emb(inputs)

print(emb.trainable_variables)
grads = tape.gradient(outputs, [inputs]+emb.trainable_variables)
dInputs, dkernel = grads
print(dkernel)
sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
sgd.apply_gradients(zip([dkernel], emb.trainable_variables))
print(emb.trainable_variables)

#print(inputs.shape, '->', outputs.shape)
#print(outputs)
#print(dInputs)
