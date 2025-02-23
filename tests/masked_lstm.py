import tensorflow as tf

emb = tf.keras.layers.Embedding(
    5, # inputDim
    4, # outputDim
    mask_zero=True,
)
lstm = tf.keras.layers.LSTM(8, return_sequences=True)

inputs = tf.Variable([
    [1,2,3,4,0,0,3,0,],
    [1,2,3,4,1,2,3,4,],
],dtype=tf.int32)

x = emb(inputs)
outputs = lstm(x)

print(inputs.shape, '->', x.shape, '->', outputs.shape)

print(outputs)
