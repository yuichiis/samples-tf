import tensorflow as tf
from functools import reduce
from operator import mul

#def layer_normalization_with_gradients(x, gamma, beta, epsilon=1e-5):
#    """
#    Layer Normalization with Gradient Computation
#
#    Args:
#      x: Input tensor (shape: [batch_size, feature_dim])
#      gamma: Scale parameter (shape: [feature_dim])
#      beta: Shift parameter (shape: [feature_dim])
#      epsilon: A small value to prevent division by zero
#
#    Returns:
#      正規化されたテンソルと勾配の辞書
#    """
#    mean = tf.math.reduce_mean(x, axis=1, keepdims=True)
#    variance = tf.math.reduce_variance(x, axis=1, keepdims=True)
#    normalized_x = (x - mean) / tf.math.sqrt(variance + epsilon)
#    scaled_x = gamma * normalized_x + beta
#
#    # 勾配計算 (連鎖律)
#    #d_scaled_x = tf.ones_like(scaled_x)  # 出力に関する勾配 (初期値は1)
#    d_scaled_x = x
#    d_gamma = tf.math.reduce_sum(d_scaled_x * normalized_x, axis=0)
#    d_beta = tf.math.reduce_sum(d_scaled_x, axis=0)
#    d_normalized_x = d_scaled_x * gamma
#    d_variance = tf.math.reduce_sum(d_normalized_x * (x - mean) * (-0.5) * (variance + epsilon)**(-1.5), axis=1, keepdims=True)
#    d_mean = tf.math.reduce_sum(d_normalized_x * (-1) / tf.math.sqrt(variance + epsilon), axis=1, keepdims=True) + \
#             tf.math.reduce_sum(d_variance * (-2) * (x - mean) / x.shape[1], axis=1, keepdims=True)
#    d_x = d_normalized_x / tf.math.sqrt(variance + epsilon) + \
#          d_variance * 2 * (x - mean) / x.shape[1] + \
#          d_mean / x.shape[1]
#
#    return scaled_x, {"d_x": d_x, "d_gamma": d_gamma, "d_beta": d_beta}
#
## 使用例
##x = tf.constant([[1, 2, 3], [4, 5, 6]])
#inputs_shape = [4*3,2]
#x = tf.reshape(tf.range(1, 1+reduce(mul, inputs_shape), dtype=tf.float32),inputs_shape)
#
#feature_dim = x.shape[1]
#gamma = tf.ones(feature_dim)
#beta = tf.zeros(feature_dim)
#normalized_x, gradients = layer_normalization_with_gradients(x, gamma, beta)
#print("Normalized x:\n", normalized_x)
#print("Gradients:\n", gradients)
#
#layernorm = tf.keras.layers.LayerNormalization()
#inputs = tf.Variable(x)
#with tf.GradientTape() as tape:
#    outputs = layernorm(inputs)
#    result = outputs * x
#
#grads = tape.gradient(result, [outputs,inputs])
#d_outputs,d_inputs = grads
#print('inputs=',inputs)
#print('outputs=',outputs)
#print('d_outputs=',d_outputs)
#print('d_inputs=',d_inputs)


###############################

#def layer_normalization_with_gradients2(x, gamma, beta, d_scaled_x, epsilon=1e-5):
#    """
#    Layer Normalization with Gradient Computation
#
#    Args:
#      x: Input tensor (shape: [batch_size, feature_dim])
#      gamma: Scale parameter (shape: [feature_dim])
#      beta: Shift parameter (shape: [feature_dim])
#      epsilon: A small value to prevent division by zero
#
#    Returns:
#      正規化されたテンソルと勾配の辞書
#    """
#    batch_size, feature_dim = tf.shape(x)[0], tf.shape(x)[1]
#    feature_dim_f = tf.cast(feature_dim, tf.float32)
#    
#    # 平均と分散を計算
#    mean = tf.math.reduce_mean(x, axis=1, keepdims=True)
#    #variance = tf.math.reduce_mean(tf.math.square(x - mean), axis=1, keepdims=True)
#    variance = tf.math.reduce_variance(x, axis=1, keepdims=True)
#
#    # 正規化
#    normalized_x = (x - mean) / tf.math.sqrt(variance + epsilon)
#    scaled_x = gamma * normalized_x + beta
#
#    # 勾配計算
#    d_scaled_x = tf.ones_like(scaled_x)  # dL/dy = 1 と仮定
#    
#    d_gamma = tf.math.reduce_sum(d_scaled_x * normalized_x, axis=0)
#    d_beta = tf.math.reduce_sum(d_scaled_x, axis=0)
#
#    d_normalized_x = d_scaled_x * gamma
#    d_variance = tf.math.reduce_sum(d_normalized_x * (x - mean) * -0.5 * tf.math.pow(variance + epsilon, -1.5), axis=1, keepdims=True)
#    d_mean = tf.math.reduce_sum(d_normalized_x * -1 / tf.math.sqrt(variance + epsilon), axis=1, keepdims=True) + d_variance * tf.math.reduce_sum(-2 * (x - mean), axis=1, keepdims=True) / feature_dim
#
#    d_x = d_normalized_x / tf.math.sqrt(variance + epsilon) + d_variance * 2 * (x - mean) / feature_dim + d_mean / feature_dim
#
#    return scaled_x, {"d_x": d_x, "d_gamma": d_gamma, "d_beta": d_beta}

#def layer_normalization_with_gradients2(x, gamma, beta, d_scaled_x, epsilon=1e-3):
#    """
#    Layer Normalization with Gradient Computation
#
#    Args:
#      x: Input tensor (shape: [batch_size, feature_dim])
#      gamma: Scale parameter (shape: [feature_dim])
#      beta: Shift parameter (shape: [feature_dim])
#      d_scaled_x: 出力の勾配 (shape: [batch_size, feature_dim])
#      epsilon: A small value to prevent division by zero
#
#    Returns:
#      正規化されたテンソルと勾配の辞書
#    """
#    batch_size, feature_dim = tf.shape(x)[0], tf.shape(x)[1]
#    feature_dim_f = tf.cast(feature_dim, tf.float32)
#
#    # 平均と分散
#    mean = tf.math.reduce_mean(x, axis=1, keepdims=True)
#    variance = tf.math.reduce_variance(x, axis=-1, keepdims=True)
#
#    # 正規化
#    center_x = x - mean
#    normalized_x = center_x / tf.math.sqrt(variance + epsilon)
#    scaled_x = gamma * normalized_x + beta
#
#    # d_gamma, d_beta
#    d_gamma = tf.math.reduce_sum(d_scaled_x * normalized_x, axis=0)
#    d_beta = tf.math.reduce_sum(d_scaled_x, axis=0)
#
#    # 逆伝播
#    d_normalized_x = d_scaled_x * gamma
#    #print('d_normalized_x',d_normalized_x)
#    # d_variance = -sum(d_normalized_x*center_x/2 * (variance + epsilon)**(-1.5))
#    d_variance = tf.math.reduce_sum(
#        d_normalized_x * center_x * -0.5 * tf.math.pow(variance + epsilon, -1.5), \
#        axis=-1, keepdims=True)
#    #print('d_variance',d_variance)
#    
#    
#    # d_mean  = sum(-d_normalized_x/sqrt(variance + epsilon)) +
#    #           d_variance * sum(-2*center_x) / feature_dim
#    d_mean = tf.math.reduce_sum( \
#            d_normalized_x * -1 / tf.math.sqrt(variance + epsilon), \
#            axis=1, keepdims=True \
#        ) \
#            + \
#        d_variance * \
#        tf.math.reduce_sum(-2 * center_x, axis=1, keepdims=True) / \
#        feature_dim_f
#    #print('d_mean',d_mean)
#
#
#    # d_x = d_normalized_x / sqrt(variance + epsilon) +
#    #       d_variance * 2 * center_x / feature_dim_f +
#    #       d_mean / feature_dim
#    tmp1 = d_normalized_x / tf.math.sqrt(variance + epsilon)
#    #print('tmp1',tmp1)
#    #print('center_x',center_x)
#    tmp2 = d_variance * 2 * center_x / feature_dim_f
#    #print('tmp2',tmp2)
#
#    d_x = tmp1 \
#            + \
#        tmp2 \
#            + \
#        d_mean / feature_dim_f
#    #print('d_x',d_x)
#
#    return scaled_x, {"d_x": d_x, "d_gamma": d_gamma, "d_beta": d_beta}


#def layer_normalization_with_gradients2(x, gamma, beta, d_scaled_x, epsilon=0.001):
#    """
#    Layer Normalization with Gradient Computation
#
#    Args:
#      x: Input tensor (shape: [batch_size, feature_dim])
#      gamma: Scale parameter (shape: [feature_dim])
#      beta: Shift parameter (shape: [feature_dim])
#      d_scaled_x: 出力の勾配 (shape: [batch_size, feature_dim])
#      epsilon: A small value to prevent division by zero
#
#    Returns:
#      正規化されたテンソルと勾配の辞書
#    """
#    batch_size, feature_dim = tf.shape(x)[0], tf.shape(x)[1]
#    feature_dim_f = tf.cast(feature_dim, tf.float32)
#
#    # 平均と分散
#    mean = tf.math.reduce_mean(x, axis=1, keepdims=True)
#    variance = tf.math.reduce_mean(tf.math.square(x - mean), axis=1, keepdims=True)
#    #variance = tf.math.reduce_variance(x, axis=-1, keepdims=True)
#
#    # 正規化
#    center_x = x - mean
#    normalized_x = center_x / (tf.math.sqrt(variance + epsilon))
#    scaled_x = gamma * normalized_x + beta
#
#    # d_gamma, d_beta
#    d_gamma = tf.math.reduce_sum(d_scaled_x * normalized_x, axis=0)
#    d_beta = tf.math.reduce_sum(d_scaled_x, axis=0)
#
#    # 逆伝播
#    d_normalized_x = d_scaled_x * gamma
#    #print('d_normalized_x',d_normalized_x)
#    # d_variance = -sum(d_normalized_x*center_x/2 * (variance + epsilon)**(-1.5))
#    d_variance = tf.math.reduce_sum(
#        d_normalized_x * center_x * -0.5 * tf.math.pow(variance + epsilon, -1.5), \
#        axis=-1, keepdims=True)
#    #print('d_variance',d_variance)
#    
#    
#    # d_mean  = sum(-d_normalized_x/sqrt(variance + epsilon)) +
#    #           d_variance * sum(-2*center_x) / feature_dim
#    d_mean = tf.math.reduce_sum( \
#            d_normalized_x * -1 / tf.math.sqrt(variance + epsilon), \
#            axis=1, keepdims=True \
#        ) \
#            + \
#        d_variance * \
#        tf.math.reduce_sum(-2 * center_x, axis=1, keepdims=True) / \
#        feature_dim_f
#    #print('d_mean',d_mean)
#
#
#    # d_x = d_normalized_x / sqrt(variance + epsilon) +
#    #       d_variance * 2 * center_x / feature_dim_f +
#    #       d_mean / feature_dim
#    tmp1 = d_normalized_x / tf.math.sqrt(variance + epsilon)
#    #print('tmp1',tmp1)
#    #print('center_x',center_x)
#    tmp2 = d_variance * 2 * center_x / feature_dim_f
#    #print('tmp2',tmp2)
#
#    d_x = tmp1 \
#            + \
#        tmp2 \
#            + \
#        d_mean / feature_dim_f
#    #print('d_x',d_x)
#
#    return scaled_x, {"d_x": d_x, "d_gamma": d_gamma, "d_beta": d_beta}


def layer_normalization_with_gradients2(x, gamma, beta, d_scaled_x, epsilon=0.001):
    """
    Layer Normalization with Gradient Computation

    Args:
      x: Input tensor (shape: [batch_size, feature_dim])
      gamma: Scale parameter (shape: [feature_dim])
      beta: Shift parameter (shape: [feature_dim])
      d_scaled_x: 出力の勾配 (shape: [batch_size, feature_dim])
      epsilon: A small value to prevent division by zero

    Returns:
      正規化されたテンソルと勾配の辞書
    """
    batch_size, feature_dim = tf.shape(x)[0], tf.shape(x)[1]
    feature_dim_f = tf.cast(feature_dim, tf.float32)

    # 平均と分散
    mean = tf.math.reduce_mean(x, axis=1, keepdims=True)
    variance = tf.math.reduce_mean(tf.math.square(x - mean), axis=1, keepdims=True)
    #variance = tf.math.reduce_variance(x, axis=-1, keepdims=True)
    std = tf.math.sqrt(variance + epsilon)

    # 正規化
    center_x = x - mean
    normalized_x = center_x / std
    scaled_x = gamma * normalized_x + beta

    # d_gamma, d_beta
    d_gamma = tf.math.reduce_sum(d_scaled_x * normalized_x, axis=0)
    d_beta = tf.math.reduce_sum(d_scaled_x, axis=0)

    # 逆伝播
    d_normalized_x = d_scaled_x * gamma
    d_center_x = d_normalized_x / std
    #print('d_normalized_x',d_normalized_x)
    # d_variance = -sum(d_normalized_x*center_x/2 * (variance + epsilon)**(-1.5))
    #d_variance = tf.math.reduce_sum(
    #    d_normalized_x * center_x * -0.5 * tf.math.pow(variance + epsilon, -1.5), \
    #    axis=-1, keepdims=True)
    #print('d_variance',d_variance)
    
    
    # d_mean  = sum(-d_normalized_x/std) +
    #           d_variance * sum(-2*center_x) / feature_dim
    #d_mean0 = tf.math.reduce_sum( \
    #        d_normalized_x * -1 / std, \
    #        axis=1, keepdims=True \
    #    ) \
    #        + \
    #    d_variance * \
    #    tf.math.reduce_sum(-2 * center_x, axis=1, keepdims=True) / \
    #    feature_dim_f
    #tmp = d_variance * \
    #    tf.math.reduce_sum(-2 * center_x, axis=1, keepdims=True) / \
    #    feature_dim_f
    #print('tmp',tmp)
    #print('d_mean0',d_mean0)

    d_mean = tf.math.reduce_sum(-d_center_x, axis=-1, keepdims=True)
    #print('d_mean',d_mean)
    d_std = tf.math.reduce_sum(-normalized_x*d_center_x, axis=-1, keepdims=True)


    # d_x = d_normalized_x / sqrt(variance + epsilon) +
    #       d_variance * 2 * center_x / feature_dim_f +
    #       d_mean / feature_dim
    #tmp1 = d_normalized_x / tf.math.sqrt(variance + epsilon)
    #print('tmp1',tmp1)
    #print('center_x',center_x)
    #tmp2 = d_variance * 2 * center_x / feature_dim_f
    #print('tmp2',tmp2)

    #d_x = tmp1 \
    #        + \
    #    tmp2 \
    #        + \
    #    d_mean / feature_dim_f
    #print('d_x',d_x)
    d_x = d_center_x + (d_mean + normalized_x*d_std) / feature_dim_f

    return scaled_x, {"d_x": d_x, "d_gamma": d_gamma, "d_beta": d_beta}


# 使用例
inputs_shape = [12, 2]
x = tf.reshape(tf.range(1, 1 + tf.math.reduce_prod(inputs_shape), dtype=tf.float32), inputs_shape)
#x = tf.constant([
#    [1.0, 2.0, 3.0],
#    [1.0, 2.0, 3.0],
#    [3.0, 2.0, 1.0],
#    [3.0, 2.0, 1.0],
#])


feature_dim = x.shape[1]
gamma = tf.ones(feature_dim)
beta = tf.zeros(feature_dim)

# テスト用の d_scaled_x
d_scaled_x = x
#d_scaled_x = tf.constant([
#    [0.0, 0.5, 1.0],
#    [0.0, 0.5, 1.0],
#    [1.0, 0.5, 0.0],
#    [1.0, 0.5, 0.0],
#])

normalized_x, gradients = layer_normalization_with_gradients2(x, gamma, beta, d_scaled_x)
#print("Normalized x:\n", normalized_x)
print("Gradients:")
for k, v in gradients.items() :
    print( k, "\n", v.numpy() )

# TensorFlow の LayerNormalization との比較
layernorm = tf.keras.layers.LayerNormalization()
inputs = tf.Variable(x)
with tf.GradientTape() as tape:
    outputs = layernorm(inputs)
    result = outputs * d_scaled_x

grads = tape.gradient(result, [outputs, inputs]+layernorm.trainable_variables)
d_outputs, d_inputs, d_gamma, d_beta = grads
#print('inputs=', inputs.numpy())
#print("outputs=\n", outputs.numpy())
#print('d_outputs=', d_outputs.numpy())
print('d_inputs=', d_inputs.numpy())
print('d_gamma=', d_gamma.numpy())
print('d_beta=', d_beta.numpy())

d_x = gradients['d_x']
print('diff(tf-custom)=',tf.math.reduce_max(tf.math.abs(normalized_x - outputs)))
print('d_diff(tf-custom)=',tf.math.reduce_max(tf.math.abs(d_x - d_inputs)))
