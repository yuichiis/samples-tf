import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def sample(logits):
  """
  ロジットからカテゴリカル分布に従ってサンプリングします。
  tfp.distributions.Categorical.sample() の代替です。

  Args:
    logits: 形状が [batch_size, num_actions] のテンソル。
            分布の正規化されていない対数確率。

  Returns:
    actions: 形状が [batch_size] のテンソル。
             サンプリングされたアクション（カテゴリのインデックス）。dtypeはint64。
  """
  # tf.random.categoricalはロジットから直接サンプリングできる
  # num_samples=1を指定すると、各バッチ項目に対して1つサンプリングする
  # 出力の形状は [batch_size, 1] になる
  samples = tf.random.categorical(logits, num_samples=1)
  
  # 形状を [batch_size, 1] から [batch_size] に変換して返す
  return tf.squeeze(samples, axis=-1)

def log_prob_entropy(logits, actions):
  """
  対数確率(log_prob)とエントロピーを計算します。
  tfp.distributions.Categorical の log_prob() と entropy() の代替です。

  Args:
    logits: 形状が [batch_size, num_actions] のテンソル。
            分布の正規化されていない対数確率。
    actions: 形状が [batch_size] のテンソル。
             対数確率を計算したいアクション（カテゴリのインデックス）。

  Returns:
    log_prob: 形状が [batch_size] のテンソル。各アクションの対数確率。
    entropy: 形状が [batch_size] のテンソル。各分布のエントロピー。
  """
  # --- 1. 対数確率 (Log Probability) の計算 ---
  # tf.nn.sparse_softmax_cross_entropy_with_logits は、指定されたaction (labels)
  # の「負の」対数確率を効率的かつ数値的に安定して計算します。
  # log_prob = -cross_entropy
  negative_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=actions, logits=logits)
  log_prob = -negative_log_prob

  # --- 2. エントロピー (Entropy) の計算 ---
  # エントロピーの定義: H(p) = - Σ_i p_i * log(p_i)
  # 数値的安定性のため、softmaxとlog_softmaxをそれぞれ使用します。
  probs = tf.nn.softmax(logits, axis=-1)
  log_probs = tf.nn.log_softmax(logits, axis=-1)
  
  # ゼロ確率の項 (p_i=0) は log(p_i) が-infになりますが、
  # p_i * log(p_i) は 0 * -inf = NaN となります。
  # しかし、softmaxの結果が厳密に0になることはまれで、
  # TFの計算では適切に処理されるため、通常は問題ありません。
  entropy = -tf.reduce_sum(probs * log_probs, axis=-1)

  return log_prob, entropy


# --- 使用例と比較検証 ---
if __name__ == "__main__":
  # バッチサイズ3、アクション数4のダミーデータを作成
  batch_size = 3
  num_actions = 4
  
  # tf.random.set_seed(42) # 結果を固定したい場合
  
  logits = tf.constant([
      [1.0, 2.0, -1.0, 0.5],   # 1番目のデータ
      [0.1, 0.2, 0.3, 4.0],   # 2番目のデータ (4番目のアクションが選ばれやすい)
      [-2.0, 1.0, -2.0, 1.0]    # 3番目のデータ (2番目と4番目が同じ確率)
  ], dtype=tf.float32)

  print("--- 自作関数のテスト ---")
  # 1. サンプリング
  sampled_actions = sample(logits)
  print(f"Logits:\n{logits.numpy()}")
  print(f"Sampled Actions: {sampled_actions.numpy()}")
  print("-" * 20)

  # 2. 対数確率とエントロピーの計算
  log_prob_val, entropy_val = log_prob_entropy(logits, sampled_actions)
  print(f"Log Probs for sampled actions:\n{log_prob_val.numpy()}")
  print(f"Entropy of the distributions:\n{entropy_val.numpy()}")
  print("\n" + "="*40 + "\n")
  
  
  print("--- TFPによる検証 ---")
  # TFPのCategorical分布を作成
  cat_dist = tfp.distributions.Categorical(logits=logits)
  
  # TFPのlog_probとentropyを計算
  # Note: TFPのsampleはシードを取れますが、ここでは比較のため同じアクションを使います
  tfp_log_prob = cat_dist.log_prob(sampled_actions)
  tfp_entropy = cat_dist.entropy()
  
  print(f"TFP Log Probs for sampled actions:\n{tfp_log_prob.numpy()}")
  print(f"TFP Entropy of the distributions:\n{tfp_entropy.numpy()}")
  print("\n" + "="*40 + "\n")
  
  
  # 結果の比較
  print("--- 結果の比較 ---")
  log_prob_diff = np.abs(log_prob_val.numpy() - tfp_log_prob.numpy()).max()
  entropy_diff = np.abs(entropy_val.numpy() - tfp_entropy.numpy()).max()
  
  print(f"Max difference in Log Prob: {log_prob_diff:.6f}")
  print(f"Max difference in Entropy:  {entropy_diff:.6f}")
  
  # 差が非常に小さいことを確認
  assert log_prob_diff < 1e-6
  assert entropy_diff < 1e-6
  print("\n✅ 自作関数の結果はTFPと一致しました。")
  