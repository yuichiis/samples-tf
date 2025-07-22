import tensorflow as tf

network = tf.keras.Sequential([
    tf.keras.layers.Dense(64, ),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(4),
])
optimizer = tf.optimizers.Adam(learning_rate=0.01)

states = tf.constant([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
targets = tf.constant([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

stateVar = tf.Variable(states)
targetVar = tf.Variable(targets)
with tf.GradientTape() as tape:
    output = network(stateVar, training=True)
    loss = tf.math.reduce_mean(tf.square(targetVar - output))

trainableVars = network.trainable_variables
grads = tape.gradient(loss, trainableVars)
print("grads count=",grads)
optimizer.apply_gradients(zip(grads, trainableVars))
