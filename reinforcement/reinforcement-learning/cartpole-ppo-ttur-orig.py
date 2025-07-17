import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym

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
    """Actor-Criticモデル"""
    def __init__(self, action_size):
        super().__init__()
        # 共通レイヤー
        self.common_layer = tf.keras.layers.Dense(128, activation='relu')
        
        # Actor: 方策（どの行動を取るか）を出力
        self.actor_head = tf.keras.layers.Dense(action_size) # 出力は行動の数 (CartPoleでは2)
        
        # Critic: 状態価値を出力
        self.critic_head = tf.keras.layers.Dense(1) # 出力はスカラー値

    def call(self, state):
        x = self.common_layer(state)
        # Actorの出力 (logits)
        action_logits = self.actor_head(x)
        # Criticの出力 (状態価値)
        value = self.critic_head(x)
        return action_logits, value

def compute_advantages_and_returns(rewards, values, dones, next_value):
    """GAEとリターンを計算する"""
    # GAE: Generalized Advantage Estimation
    # A_t = delta_t + gamma * lambda * A_{t+1}
    # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0
    
    # 最後のステップから逆順に計算していく
    for t in reversed(range(len(rewards))):
        if dones[t]:
            # エピソードが終了した場合
            delta = rewards[t] - values[t]
            last_advantage = delta
        else:
            delta = rewards[t] + GAMMA * values[t+1] - values[t]
            last_advantage = delta + GAMMA * GAE_LAMBDA * last_advantage
        advantages[t] = last_advantage
        
    # リターン = アドバンテージ + 状態価値
    returns = advantages + values[:-1] # next_valueは不要なため[:-1]
    
    # アドバンテージの正規化 (学習を安定させるため)
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    
    return advantages, returns

def main():
    # 環境の初期化
    env = gym.make(ENV_NAME)
    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    # モデルとオプティマイザの初期化
    model = ActorCritic(action_size)
    # 修正前
    # optimizer = tf.keras.optimizers.Adam(learning_rate=ACTOR_LR) # Actor/Criticまとめて更新
    # 修正後
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=ACTOR_LR)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=CRITIC_LR) # ここでCRITIC_LRを使用

    # メインの学習ループ
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

        state = env.reset()
        
        for t in range(N_ROLLOUT_STEPS):
            total_step += 1
            state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
            
            # 現在の方策に基づいて行動を決定
            action_logits, value = model(state_tensor)
            dist = tfp.distributions.Categorical(logits=action_logits)
            action = dist.sample()[0].numpy()
            log_prob = dist.log_prob(action)
            
            # 環境中で行動を実行
            next_state, reward, done, _ = env.step(action)
            
            # データを保存
            states_mem.append(state)
            actions_mem.append(action)
            rewards_mem.append(reward)
            dones_mem.append(done)
            values_mem.append(value[0,0].numpy())
            log_probs_mem.append(log_prob[0].numpy())
            
            state = next_state
            if done:
                state = env.reset()

        # 最後の状態の価値を計算
        next_state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        _, next_value = model(next_state_tensor)
        values_mem.append(next_value[0,0].numpy())

        # === 2. アドバンテージとリターンの計算 ===
        advantages, returns = compute_advantages_and_returns(
            rewards_mem, values_mem, dones_mem, next_value
        )
        
        # NumPy配列をTensorに変換
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
                
                # 修正前：1つの損失でまとめて更新
                # with tf.GradientTape() as tape:
                #     # 現在のモデルで予測
                #     new_logits, new_values = model(states_b)
                #     new_values = tf.squeeze(new_values)
                #     
                #     new_dist = tfp.distributions.Categorical(logits=new_logits)
                #     new_log_probs = new_dist.log_prob(actions_b)
                #     
                #     # 1. Actor Loss (Policy Loss)
                #     ratio = tf.exp(new_log_probs - old_log_probs_b)
                #     clipped_ratio = tf.clip_by_value(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
                #     actor_loss = -tf.reduce_mean(
                #         tf.minimum(ratio * advantages_b, clipped_ratio * advantages_b)
                #     )
                #     
                #     # 2. Critic Loss (Value Loss)
                #     critic_loss = tf.reduce_mean(tf.square(returns_b - new_values))
                #     
                #     # 3. Entropy Loss (探索を促進)
                #     entropy_loss = -tf.reduce_mean(new_dist.entropy())
                #     
                #     # Total Loss
                #     # 係数は一般的な値を使用
                #     total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
                #     
                # # 勾配を計算してモデルを更新
                # grads = tape.gradient(total_loss, model.trainable_variables)
                # optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # 修正後：ActorとCriticで別々に更新
                with tf.GradientTape(persistent=True) as tape:
                    # 現在のモデルで予測
                    new_logits, new_values = model(states_b)
                    new_values = tf.squeeze(new_values)
                    new_dist = tfp.distributions.Categorical(logits=new_logits)

                    # --- 損失計算 ---
                    # Actor Loss (方策の損失)
                    new_log_probs = new_dist.log_prob(actions_b)
                    ratio = tf.exp(new_log_probs - old_log_probs_b)
                    clipped_ratio = tf.clip_by_value(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
                    actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages_b, clipped_ratio * advantages_b))

                    # Entropy Loss (探索を促進)
                    entropy_loss = -tf.reduce_mean(new_dist.entropy())

                    # Actorの最終的な損失 (エントロピー項も方策に関するものなので含める)
                    actor_total_loss = actor_loss + 0.01 * entropy_loss

                    # Critic Loss (価値関数の損失)
                    critic_loss = tf.reduce_mean(tf.square(returns_b - new_values))

                # --- 勾配の計算と適用 ---
                # 1. Actorの更新
                actor_vars = model.common_layer.trainable_variables + model.actor_head.trainable_variables
                actor_grads = tape.gradient(actor_total_loss, actor_vars)
                actor_optimizer.apply_gradients(zip(actor_grads, actor_vars))

                # 2. Criticの更新
                critic_vars = model.common_layer.trainable_variables + model.critic_head.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_vars)
                critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))

                # persistent tapeを使った後はメモリを解放する
                del tape

        # === 4. 進捗の評価と表示 ===
        eval_env = gym.make(ENV_NAME)
        scores = []
        for _ in range(10): # 10回試行して平均スコアを計算
            state = eval_env.reset()
            done = False
            score = 0
            while not done:
                state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
                action_logits, _ = model(state_tensor)
                # 評価時は最も確率の高い行動を選択
                action = tf.argmax(action_logits, axis=1)[0].numpy() 
                state, reward, done, _ = eval_env.step(action)
                score += reward
            scores.append(score)
        avg_score = np.mean(scores)
        
        print(f"Episode: {episode_count}, Total Steps: {total_step}, Avg Score: {avg_score:.2f}")

        if avg_score >= TARGET_SCORE:
            print(f"\n目標スコア {TARGET_SCORE} を達成しました！学習を終了します。")
            # 学習済みモデルの保存 (任意)
            # model.save_weights('ppo-cartpole.weights.h5')
            break
            
if __name__ == '__main__':
    main()
