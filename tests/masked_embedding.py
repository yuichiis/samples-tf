import tensorflow as tf

emb = tf.keras.layers.Embedding(
    5, # inputDim
    4, # outputDim
    #mask_zero=True,
)

inputs = tf.Variable([
    [1,2,3,4,0,0,0,0,],
    [1,2,3,4,1,2,3,4,],
],dtype=tf.int32)

outputs = emb(inputs)

print(inputs.shape, '->', outputs.shape)

print(outputs)
