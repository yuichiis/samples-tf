import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # INFOとWARNINGログを抑制し、OUT_OF_RANGEの表示を消す
import tensorflow as tf
import numpy as np
import gymnasium as gym

# ===================================================================
# TensorFlow Probabilityの代替となるヘルパー関数 (変更なし)
# ===================================================================

def sample(logits):
  """
  ロジットからカテゴリカル分布に従ってサンプリングします。
  tfp.distributions.Categorical.sample() の代替です。
  """
  samples = tf.random.categorical(logits, num_samples=1)
  return tf.squeeze(samples, axis=-1)

def log_prob_entropy(logits, actions):
  """
  対数確率(log_prob)とエントロピーを計算します。
  tfp.distributions.Categorical の log_prob() と entropy() の代替です。
  """
  negative_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=actions, logits=logits)
  log_prob = -negative_log_prob
  
  probs = tf.nn.softmax(logits, axis=-1)
  log_probs = tf.nn.log_softmax(logits, axis=-1)
  entropy = -tf.reduce_sum(probs * log_probs, axis=-1)

  return log_prob, entropy


# ===================================================================
# PPOの実装
# ===================================================================

# ハイパーパラメータの設定
ENV_NAME = 'CartPole-v1'
GAMMA = 0.99  # 割引率
GAE_LAMBDA = 0.95 # GAEのλ
ACTOR_LR = 1e-4   # Actorの学習率
CRITIC_LR = 3e-4  # Criticの学習率
CLIP_EPSILON = 0.2 # PPOのクリッピングパラメータ
N_EPOCHS = 10     # 1回のデータ収集あたりの学習エポック数
BATCH_SIZE = 64   # ミニバッチサイズ
N_ROLLOUT_STEPS = 2048 # 1回のデータ収集で実行するステップ数
TARGET_SCORE = 475 # 学習を終了する目標スコア

class ActorCritic(tf.keras.Model):
    """Actor-Criticモデル (変更なし)"""
    def __init__(self, action_size):
        super().__init__()
        self.common_layer1 = tf.keras.layers.Dense(128, activation='relu')
        self.common_layer2 = tf.keras.layers.Dense(128, activation='relu')
        self.actor_head = tf.keras.layers.Dense(action_size)
        self.critic_head = tf.keras.layers.Dense(1)

    def call(self, state):
        x = self.common_layer1(state)
        x = self.common_layer2(x)
        action_logits = self.actor_head(x)
        value = self.critic_head(x)
        return action_logits, value

# -------------------------------------------------------------------- #
# GAE計算関数のシグネチャを修正
# -------------------------------------------------------------------- #
def compute_advantages_and_returns(rewards, values, dones):
    """
    GAEとリターンを計算する。
    Args:
        rewards (list): ステップごとの報酬のリスト (長さN_ROLLOUT_STEPS)
        values (np.ndarray): ステップごとの状態価値の配列 (長さN_ROLLOUT_STEPS + 1)。
                             最後の要素はnext_stateの価値。
        dones (list): ステップごとの終了フラグのリスト (長さN_ROLLOUT_STEPS)
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0
    
    # 最後のステップから逆順に計算していく
    for t in reversed(range(len(rewards))):
        if dones[t]:
            delta = rewards[t] - values[t]
            last_advantage = delta
        else:
            # values[t+1] は t番目のステップにおける次状態の価値 V(s_{t+1})
            delta = rewards[t] + GAMMA * values[t+1] - values[t]
            last_advantage = delta + GAMMA * GAE_LAMBDA * last_advantage
        advantages[t] = last_advantage
        
    # リターン = アドバンテージ + 状態価値
    # values は長さ N+1 なので、最後の要素を除いて長さを N に合わせる
    returns = advantages + values[:-1]
    
    # アドバンテージの正規化
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    
    return advantages, returns

def main():
    env = gym.make(ENV_NAME)
    action_size = env.action_space.n

    model = ActorCritic(action_size)
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=ACTOR_LR)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=CRITIC_LR)

    episode_count = 0
    total_step = 0
    
    while True:
        episode_count += 1
        
        # === 1. データ収集 (Rollout) ===
        # valuesとlog_probsは保存しない
        states_mem = []
        actions_mem = []
        rewards_mem = []
        dones_mem = []

        state, _ = env.reset()
        
        for t in range(N_ROLLOUT_STEPS):
            total_step += 1
            state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
            
            # 行動をサンプリングする（valueとlog_probはここでは不要）
            action_logits, _ = model(state_tensor)
            action_tensor = sample(action_logits)
            action = action_tensor.numpy()[0]
            
            next_state, reward, done, truncated, _ = env.step(action)
            
            # 基本的なデータのみ保存
            states_mem.append(state)
            actions_mem.append(action)
            rewards_mem.append(reward)
            dones_mem.append(done or truncated)
            
            state = next_state
            if done or truncated:
                state, _ = env.reset()

        # === 2. 学習データの準備 ===
        
        # NumPy配列をTensorに変換
        states_tensor = tf.convert_to_tensor(np.array(states_mem), dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(np.array(actions_mem), dtype=tf.int32)
        
        # -------------------------------------------------------------------- #
        # ここで values と log_probs を再計算する (ここが重要な変更点)
        # -------------------------------------------------------------------- #
        # 「データ収集に使った（＝更新前の）モデル」を使って一括で計算する
        # tf.GradientTapeの外で実行することで、勾配計算の対象外とする
        
        # (1) 古い方策での対数確率 (old_log_probs) と状態価値 (values) を計算
        logits_old, values_old_tensor = model(states_tensor)
        log_probs_old_tensor, _ = log_prob_entropy(logits_old, actions_tensor)

        # (2) 最後の状態の価値 (next_value) を計算
        next_state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        _, next_value_tensor = model(next_state_tensor)
        
        # (3) GAE計算のために、状態価値をNumPy配列にまとめる
        # 長さは N_ROLLOUT_STEPS + 1 になる
        values_for_gae = np.append(tf.squeeze(values_old_tensor).numpy(), tf.squeeze(next_value_tensor).numpy())
        
        # -------------------------------------------------------------------- #
        
        # アドバンテージとリターンを計算
        advantages, returns = compute_advantages_and_returns(
            rewards_mem, values_for_gae, dones_mem
        )
        
        # NumPy配列をTensorに変換
        advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)

        # === 3. モデルの学習 ===
        dataset = tf.data.Dataset.from_tensor_slices((
            # 再計算した log_probs_old_tensor をデータセットに含める
            states_tensor, actions_tensor, log_probs_old_tensor, advantages_tensor, returns_tensor
        ))
        dataset = dataset.shuffle(buffer_size=N_ROLLOUT_STEPS).batch(BATCH_SIZE)

        for _ in range(N_EPOCHS):
            for batch in dataset:
                states_b, actions_b, old_log_probs_b, advantages_b, returns_b = batch
                
                with tf.GradientTape(persistent=True) as tape:
                    # 現在の（更新対象の）モデルで予測
                    new_logits, new_values = model(states_b)
                    new_values = tf.squeeze(new_values)
                    
                    # 新しい方策での対数確率とエントロピーを計算
                    new_log_probs, entropy = log_prob_entropy(new_logits, actions_b)

                    # Actor Loss
                    ratio = tf.exp(new_log_probs - old_log_probs_b)
                    clipped_ratio = tf.clip_by_value(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
                    actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages_b, clipped_ratio * advantages_b))

                    # Entropy Loss
                    entropy_loss = -tf.reduce_mean(entropy)
                    
                    actor_total_loss = actor_loss + 0.01 * entropy_loss

                    # Critic Loss
                    critic_loss = tf.reduce_mean(tf.square(returns_b - new_values))

                # Actorの更新
                actor_vars = model.common_layer.trainable_variables + model.actor_head.trainable_variables
                actor_grads = tape.gradient(actor_total_loss, actor_vars)
                actor_optimizer.apply_gradients(zip(actor_grads, actor_vars))

                # Criticの更新
                critic_vars = model.common_layer.trainable_variables + model.critic_head.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_vars)
                critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))

                del tape

        # === 4. 進捗の評価と表示 ===
        eval_env = gym.make(ENV_NAME)
        scores = []
        for _ in range(10):
            state, _ = eval_env.reset()
            done, truncated = False, False
            score = 0
            while not (done or truncated):
                state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
                action_logits, _ = model(state_tensor)
                action = tf.argmax(action_logits, axis=1)[0].numpy() 
                state, reward, done, truncated, _ = eval_env.step(action)
                score += reward
            scores.append(score)
        avg_score = np.mean(scores)
        
        print(f"Episode: {episode_count}, Total Steps: {total_step}, Avg Score: {avg_score:.2f}")

        if avg_score >= TARGET_SCORE:
            print(f"\n目標スコア {TARGET_SCORE} を達成しました！学習を終了します。")
            break
            
if __name__ == '__main__':
    main()