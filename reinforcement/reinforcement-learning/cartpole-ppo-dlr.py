import tensorflow as tf
# import tensorflow_probability as tfp # tensorflow_probabilityは不要になるので削除
import numpy as np
import gym

# ハイパーパラメータの設定 (変更なし)
ENV_NAME = 'CartPole-v1'
GAMMA = 0.99
GAE_LAMBDA = 0.95
ACTOR_LR = 1e-4
CRITIC_LR = 3e-4
CLIP_EPSILON = 0.2
N_EPOCHS = 10
BATCH_SIZE = 64
N_ROLLOUT_STEPS = 2048
TARGET_SCORE = 475

class ActorCritic(tf.keras.Model):
    """Actor-Criticモデル (変更なし)"""
    def __init__(self, action_size):
        super().__init__()
        self.common_layer = tf.keras.layers.Dense(128, activation='relu')
        self.actor_head = tf.keras.layers.Dense(action_size)
        self.critic_head = tf.keras.layers.Dense(1)

    def call(self, state):
        x = self.common_layer(state)
        action_logits = self.actor_head(x)
        value = self.critic_head(x)
        return action_logits, value

def compute_advantages_and_returns(rewards, values, dones, next_value):
    """GAEとリターンを計算する (変更なし)"""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if dones[t]:
            delta = rewards[t] - values[t]
            last_advantage = delta
        else:
            delta = rewards[t] + GAMMA * values[t+1] - values[t]
            last_advantage = delta + GAMMA * GAE_LAMBDA * last_advantage
        advantages[t] = last_advantage
        
    returns = advantages + values[:-1]
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    
    return advantages, returns

def main():
    env = gym.make(ENV_NAME)
    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    model = ActorCritic(action_size)
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=ACTOR_LR)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=CRITIC_LR)

    episode_count = 0
    total_step = 0
    
    while True:
        episode_count += 1
        
        # === 1. データ収集 (Rollout) ===
        states_mem = []
        actions_mem = []
        rewards_mem = []
        dones_mem = []
        values_mem = []
        log_probs_mem = []

        state, _ = env.reset()
        
        for t in range(N_ROLLOUT_STEPS):
            total_step += 1
            state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
            
            # --- TFPからの変更点 (データ収集) ---
            action_logits, value = model(state_tensor)
            
            # 元のコード:
            # dist = tfp.distributions.Categorical(logits=action_logits)
            # action = dist.sample()[0].numpy()
            # log_prob = dist.log_prob(action)
            
            # 変更後:
            # 1. 行動をサンプリング
            # tf.random.categoricalはロジットからインデックスをサンプリングする
            action_tensor = tf.random.categorical(action_logits, 1)[0, 0]
            action = action_tensor.numpy()
            
            # 2. その行動の対数確率を計算
            # まず全ての行動の対数確率を計算
            log_probs_all = tf.nn.log_softmax(action_logits)
            # サンプリングした行動に対応する対数確率を取得
            log_prob = log_probs_all[0, action]
            # --- 変更点ここまで ---

            next_state, reward, done, truncated, _ = env.step(action)
            
            states_mem.append(state)
            actions_mem.append(action)
            rewards_mem.append(reward)
            dones_mem.append(done)
            values_mem.append(value[0,0].numpy())
            # 元のコードではlog_probが(1,)のTensorだったので[0]でアクセスしていたが、
            # 今回はスカラーTensorなのでそのままnumpy()に変換
            log_probs_mem.append(log_prob.numpy())
            
            state = next_state
            if done or truncated:
                state, _ = env.reset()

        next_state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        _, next_value = model(next_state_tensor)
        values_mem.append(next_value[0,0].numpy())

        # === 2. アドバンテージとリターンの計算 (変更なし) ===
        advantages, returns = compute_advantages_and_returns(
            rewards_mem, values_mem, dones_mem, next_value
        )
        
        states_tensor = tf.convert_to_tensor(np.array(states_mem), dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(np.array(actions_mem), dtype=tf.int32)
        log_probs_tensor = tf.convert_to_tensor(np.array(log_probs_mem), dtype=tf.float32)
        advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)

        # === 3. モデルの学習 ===
        dataset = tf.data.Dataset.from_tensor_slices((
            states_tensor, actions_tensor, log_probs_tensor, advantages_tensor, returns_tensor
        ))
        dataset = dataset.shuffle(buffer_size=N_ROLLOUT_STEPS).batch(BATCH_SIZE)

        for _ in range(N_EPOCHS):
            for batch in dataset:
                states_b, actions_b, old_log_probs_b, advantages_b, returns_b = batch
                
                with tf.GradientTape(persistent=True) as tape:
                    new_logits, new_values = model(states_b)
                    new_values = tf.squeeze(new_values)
                    
                    # --- TFPからの変更点 (損失計算) ---
                    # 元のコード:
                    # new_dist = tfp.distributions.Categorical(logits=new_logits)
                    # new_log_probs = new_dist.log_prob(actions_b)
                    # entropy = new_dist.entropy()

                    # 変更後:
                    # 1. 新しい方策での対数確率を計算
                    log_probs_all = tf.nn.log_softmax(new_logits)
                    indices = tf.stack([tf.range(tf.shape(actions_b)[0]), actions_b], axis=1)
                    new_log_probs = tf.gather_nd(log_probs_all, indices)
                    
                    # 2. エントロピーを計算 H(p) = -Σ p*log(p)
                    probs = tf.nn.softmax(new_logits)
                    entropy_per_sample = -tf.reduce_sum(probs * log_probs_all, axis=1)
                    entropy = entropy_per_sample
                    # --- 変更点ここまで ---

                    # --- 損失計算 (TFPを使わない以外は変更なし) ---
                    # Actor Loss (方策の損失)
                    ratio = tf.exp(new_log_probs - old_log_probs_b)
                    clipped_ratio = tf.clip_by_value(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
                    actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages_b, clipped_ratio * advantages_b))

                    # Entropy Loss (探索を促進)
                    entropy_loss = -tf.reduce_mean(entropy)

                    # Actorの最終的な損失
                    actor_total_loss = actor_loss + 0.01 * entropy_loss

                    # Critic Loss (価値関数の損失)
                    critic_loss = tf.reduce_mean(tf.square(returns_b - new_values))

                # --- 勾配の計算と適用 (変更なし) ---
                actor_vars = model.common_layer.trainable_variables + model.actor_head.trainable_variables
                actor_grads = tape.gradient(actor_total_loss, actor_vars)
                actor_optimizer.apply_gradients(zip(actor_grads, actor_vars))

                critic_vars = model.common_layer.trainable_variables + model.critic_head.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_vars)
                critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))
                
                del tape

        # === 4. 進捗の評価と表示 (変更なし) ===
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
