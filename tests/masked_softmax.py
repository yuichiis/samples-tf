import tensorflow as tf

class MaskedSoftmax(tf.keras.layers.Layer):
    def call(self, inputs, mask=None):
        if mask is not None:
            # マスクされていない要素のみでソフトマックスを計算
            mask = tf.cast(mask, tf.float32)
            masked_inputs = inputs * mask
            exp_x = tf.exp(masked_inputs - tf.reduce_max(masked_inputs, axis=-1, keepdims=True)) * mask
            return exp_x / tf.reduce_sum(exp_x, axis=-1, keepdims=True)
        else:
            return tf.keras.layers.Softmax()(inputs)

# 使用例
masked_softmax = MaskedSoftmax()

a = tf.constant([
    [[1., 2., 3., 4.],
     [1., 2., 3., 4.],
     [1., 2., 3., 4.]],
    [[1., 2., 3., 4.],
     [1., 2., 3., 4.],
     [1., 2., 3., 4.]],
])
mask = tf.constant([
    [True,False,False,False],
    [True,True,False,False],
    [True,True,True,True],
])

print('===== native softmax =====')
y = a * tf.keras.layers.Softmax()(a, mask=mask)
print(y)

va = tf.Variable(a)
#v1 = tf.Variable(tf.ones([2,3,4]))
print('===== custom softmax =====')
with tf.GradientTape() as tape:
    y = a * masked_softmax(va, mask=mask)
#    y = a * v1

print(y)
print('===== grads =====')
grads = tape.gradient(y, va)
#grads = tape.gradient(y, v1)
print(grads)
