import numpy as np
import tensorflow as tf # 検証用

def log_softmax_forward(x):
    """
    NumPyによるlog_softmaxの順方向計算。
    """
    dtype = x.dtype
    max_x = np.max(x, axis=-1, keepdims=True)
    x_shifted = x - max_x
    exp_shifted = np.exp(x_shifted)
    sum_exp_shifted = np.sum(exp_shifted, axis=-1, keepdims=True)
    log_sum_exp = np.log(sum_exp_shifted)
    logsumexp_term = max_x + log_sum_exp
    out = x - logsumexp_term
    
    softmax_out = np.exp(out).astype(dtype)
    cache = (softmax_out,)
    
    return out, cache

def log_softmax_backward(dout, cache):
    """
    NumPyによるlog_softmaxの逆方向計算（逆伝播）。
    """
    softmax_out, = cache
    
    sum_dout = np.sum(dout, axis=-1, keepdims=True)
    dx = dout - softmax_out * sum_dout
    
    return dx

# --- 動作確認 ---

# サンプルデータを作成
input_data = np.array([[1.0, 2.0, 3.0, 4.0],
                       [10.0, 5.0, 1.0, -5.0]], dtype=np.float32)

# === 1. 順方向の検証 ===
print("--- 1. 順方向（Forward Pass）の検証 ---")
numpy_result, cache = log_softmax_forward(input_data)
tf_result = tf.nn.log_softmax(input_data, axis=-1).numpy()
print("入力データ:\n", input_data)
print("\nNumPyによる自作関数の結果:\n", numpy_result)
print("\nTensorFlowの結果:\n", tf_result)
is_forward_close = np.allclose(numpy_result, tf_result, atol=1e-6)
print(f"\n順方向の結果は一致していますか？: {is_forward_close}")
print("\n" + "="*50 + "\n")

# === 2. 逆方向（Backward Pass）の検証 ===
print("--- 2. 逆方向（Backward Pass）の検証 ---")

np.random.seed(0)
dout = np.random.randn(*input_data.shape).astype(np.float32)

dx_analytical = log_softmax_backward(dout, cache)

# 数値微分ではfloat64を使用する
input_data_f64 = input_data.astype(np.float64)
dout_f64 = dout.astype(np.float64) # doutもfloat64に変換

# ★★★★★ ここが最重要修正点 ★★★★★
def f(x):
    """数値微分用の損失関数シミュレーション。一貫して高精度(float64)で計算する。"""
    # x (float64) をそのまま float32 に変換せずに使う
    log_softmax_val, _ = log_softmax_forward(x)
    # dout も float64 版を使って計算する
    return np.sum(log_softmax_val * dout_f64)

epsilon = 1e-7
dx_numerical = np.zeros_like(input_data_f64)

it = np.nditer(input_data_f64, flags=['multi_index'], op_flags=['readwrite'])
while not it.finished:
    idx = it.multi_index
    original_val = input_data_f64[idx]
    
    input_data_f64[idx] = original_val + epsilon
    fx_plus_h = f(input_data_f64)
    
    input_data_f64[idx] = original_val - epsilon
    fx_minus_h = f(input_data_f64)
    
    dx_numerical[idx] = (fx_plus_h - fx_minus_h) / (2 * epsilon)
    
    input_data_f64[idx] = original_val
    it.iternext()

print("上流からの勾配 (dout):\n", dout)
print("\n自作逆伝播関数による勾配 (dx_analytical):\n", dx_analytical)
# 比較のために数値微分の結果をfloat32に戻して表示
print("\n数値微分による勾配 (dx_numerical):\n", dx_numerical.astype(np.float32))

numerator = np.linalg.norm(dx_analytical - dx_numerical)
denominator = np.linalg.norm(dx_analytical) + np.linalg.norm(dx_numerical)
relative_error = numerator / (denominator + 1e-12)

print(f"\n相対誤差: {relative_error:.10f}")
is_backward_close = relative_error < 1e-7
print(f"逆方向の勾配は一致していますか？: {is_backward_close}")