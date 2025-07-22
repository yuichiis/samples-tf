import tensorflow as tf

x = tf.Variable([
    [1.,2.],
    [3.,4.],
])

with tf.GradientTape() as tape:
    y0 = x + 1.
    y1 = x + 1.
    z = y0 + y1

dx = tape.gradient(z, x)
print(dx)
