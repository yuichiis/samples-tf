import tensorflow as tf
import numpy as np

def custom_log_softmax(x):
    """
    指定された数値的に安定な式を用いてlog_softmaxを計算します。
    
    log_softmax(x_i) = x_i - (max(x) + log(sum(exp(x_j - max(x)))))
    
    Args:
        x: 入力テンソル (tf.Tensor or array-like)。
           log_softmaxは最後の軸に沿って計算されます。
           
    Returns:
        log_softmax計算後のテンソル (tf.Tensor)。
    """
    # 入力をTensorFlowのTensorに変換
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    
    # 1. max(x)を計算
    #    最後の軸に沿って最大値を見つける
    #    keepdims=Trueにすることで、後のブロードキャストが容易になる
    max_x = tf.reduce_max(x, axis=-1, keepdims=True)
    
    # 2. log(sum(exp(x_j - max(x)))) の部分を計算 (LogSumExp)
    #    x - max_x
    x_shifted = x - max_x
    #    exp(x - max_x)
    exp_shifted = tf.exp(x_shifted)
    #    sum(exp(x - max_x))
    sum_exp_shifted = tf.reduce_sum(exp_shifted, axis=-1, keepdims=True)
    #    log(sum(exp(x - max_x)))
    log_sum_exp = tf.math.log(sum_exp_shifted)
    
    # 3. logsumexpの項全体を計算
    #    max(x) + log(sum(exp(x_j - max(x))))
    logsumexp_term = max_x + log_sum_exp
    
    # 4. 最終的な式: x_i - logsumexp_term
    log_softmax_output = x - logsumexp_term
    
    return log_softmax_output

# --- 動作確認 ---
# サンプルデータを作成
# 2バッチ、各バッチに4つのクラスを持つロジットを想定
input_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0],
                            [10.0, 5.0, 1.0, -5.0]], dtype=tf.float32)

# 自作関数で計算
custom_result = custom_log_softmax(input_tensor)

# TensorFlowの組み込み関数で計算
tf_result = tf.nn.log_softmax(input_tensor, axis=-1)

# 結果の表示
print("--- 入力テンソル ---")
print(input_tensor.numpy())
print("\n" + "="*30 + "\n")

print("--- 自作のlog_softmaxの結果 ---")
print(custom_result.numpy())
print("\n" + "="*30 + "\n")

print("--- tf.nn.log_softmaxの結果 ---")
print(tf_result.numpy())
print("\n" + "="*30 + "\n")

# 結果がほぼ一致するかを確認
difference = tf.reduce_sum(tf.abs(custom_result - tf_result))
print(f"自作関数とTF組み込み関数の差の合計: {difference.numpy():.10f}")

is_close = np.allclose(custom_result.numpy(), tf_result.numpy())
print(f"結果は一致していますか？: {is_close}")

# オーバーフローが起こりうる大きな値でテスト
large_input_tensor = tf.constant([[1000.0, 1001.0, 999.0]], dtype=tf.float32)
print("\n--- 大きな値でのテスト ---")
print("入力:", large_input_tensor.numpy())
print("自作関数の結果:", custom_log_softmax(large_input_tensor).numpy())
print("TF関数の結果:", tf.nn.log_softmax(large_input_tensor).numpy())