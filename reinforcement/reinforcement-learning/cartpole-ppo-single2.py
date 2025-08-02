import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # INFOとWARNINGログを抑制

import tensorflow as tf
import numpy as np
import gymnasium as gym
import imageio

# ===================================================================
# ヘルパー関数 (変更なし)
# ===================================================================

def sample(logits):
    samples = tf.random.categorical(logits, num_samples=1)
    return tf.squeeze(samples, axis=-1)

#def log_prob_entropy(logits, actions):
#    negative_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
#        labels=actions, logits=logits)
#    log_prob = -negative_log_prob
#  
#    probs = tf.nn.softmax(logits, axis=-1)
#    log_probs = tf.nn.log_softmax(logits, axis=-1)
#    entropy = -tf.reduce_sum(probs * log_probs, axis=-1)
#    
#    return log_prob, entropy

def log_prob_entropy(logits, actions):
    # 1. 新しい方策での対数確率を計算
    #log_probs_all = tf.nn.log_softmax(logits)
    log_probs_all = tf.math.log(tf.nn.softmax(logits))
    indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
    new_log_probs = tf.gather_nd(log_probs_all, indices)

    # 2. エントロピーを計算 H(p) = -Σ p*log(p)
    probs = tf.nn.softmax(logits)
    entropy_per_sample = -tf.reduce_sum(probs * log_probs_all, axis=1)
    entropy = entropy_per_sample

    return new_log_probs, entropy

def standardize(
    x,         # (rolloutSteps)
    ddof=None,
    ) :
    # baseline
    mean = np.mean(x)     # ()

    baseX = x - mean                    # (rolloutSteps)
    # std
    if ddof:
        n = len(x)-1
    else :
        n = len(x)

    variance = np.sum(np.square(baseX)) / n                 # ()
    stdDev = np.sqrt(variance)                              # ()
    # standardize
    result = baseX / (stdDev + 1e-8)                        # (rolloutSteps)
    return result                                           # (rolloutSteps)


# ===================================================================
# PPOの実装
# ===================================================================

# -------------------------------------------------------------------- #
# ハイパーパラメータを修正
# -------------------------------------------------------------------- #
ENV_NAME = 'CartPole-v1'
GAMMA = 0.99
GAE_LAMBDA = 0.95
# ACTOR_LR と CRITIC_LR を1つの学習率に統合
LEARNING_RATE = 3e-4 # 一般的なAdamの学習率
CLIP_EPSILON = 0.2
N_EPOCHS = 10
BATCH_SIZE = 64
N_ROLLOUT_STEPS = 2048
TARGET_SCORE = 475
VALUE_LOSS_WEIGHT = 0.5
ENTROPY_WEIGHT = 0.01
STANDARDIZE = True


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

def compute_advantages_and_returns(rewards, values, dones):
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
    
    return advantages, returns

def get_best_action(model, state):
    logits, _ = model(state.reshape((1, -1)))
    return tf.argmax(logits[0]).numpy()

def main():
    env = gym.make(ENV_NAME)
    action_size = env.action_space.n

    model = ActorCritic(action_size)

    # -------------------------------------------------------------------- #
    # オプティマイザを1つに統合
    # -------------------------------------------------------------------- #
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    episode_count = 0
    total_step = 0
    episode_score = 0
    last_episode_scores = []
    
    while True:
        episode_count += 1
        avg_loss = 0
        avg_entropy = 0
        
        # === 1. データ収集 (Rollout) (変更なし) ===
        states_mem, actions_mem, rewards_mem, dones_mem = [], [], [], []
        state, _ = env.reset()
        
        for t in range(N_ROLLOUT_STEPS):
            total_step += 1
            state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
            
            action_logits, _ = model(state_tensor)
            action = sample(action_logits).numpy()[0]
            
            next_state, reward, done, truncated, _ = env.step(action)
            episode_score += reward
            
            states_mem.append(state)
            actions_mem.append(action)
            rewards_mem.append(reward)
            dones_mem.append(done or truncated)
            
            state = next_state
            if done or truncated:
                state, _ = env.reset()
                last_episode_scores.append(episode_score)
                episode_score = 0

        # === 2. 学習データの準備 (変更なし) ===
        states_tensor = tf.convert_to_tensor(np.array(states_mem), dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(np.array(actions_mem), dtype=tf.int32)
        
        logits_old, values_old_tensor = model(states_tensor)
        log_probs_old_tensor, _ = log_prob_entropy(logits_old, actions_tensor)

        next_state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        _, next_value_tensor = model(next_state_tensor)
        
        values_for_gae = np.append(tf.squeeze(values_old_tensor).numpy(), tf.squeeze(next_value_tensor).numpy())
        
        advantages, returns = compute_advantages_and_returns(
            rewards_mem, values_for_gae, dones_mem
        )

        if STANDARDIZE:
            #advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            advantages = standardize(advantages)

        advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices((
            states_tensor, actions_tensor, log_probs_old_tensor, advantages_tensor, returns_tensor
        ))
        dataset = dataset.shuffle(buffer_size=N_ROLLOUT_STEPS).batch(BATCH_SIZE)

        # === 3. モデルの学習 ===
        # -------------------------------------------------------------------- #
        # 損失計算と勾配更新を統合
        # -------------------------------------------------------------------- #
        num_batches = 0
        for _ in range(N_EPOCHS):
            for batch in dataset:
                states_b, actions_b, old_log_probs_b, advantages_b, returns_b = batch
                # persistent=True は不要
                with tf.GradientTape() as tape:
                    # 現在のモデルで予測
                    new_logits, new_values = model(states_b)
                    new_values = tf.squeeze(new_values)
                    
                    new_log_probs, entropy = log_prob_entropy(new_logits, actions_b)

                    # 1. Actor Loss (Policy Loss)
                    ratio = tf.exp(new_log_probs - old_log_probs_b)
                    clipped_ratio = tf.clip_by_value(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
                    actor_loss = -tf.reduce_mean(
                        tf.minimum(ratio * advantages_b, clipped_ratio * advantages_b)
                    )
                    
                    # 2. Critic Loss (Value Loss)
                    critic_loss = tf.reduce_mean(tf.square(returns_b - new_values))
                    
                    # 3. Entropy Loss (探索を促進)
                    entropy_loss = -tf.reduce_mean(entropy)
                    
                    # 3つの損失を足し合わせて合計損失を計算
                    # 係数はPPOで一般的に使われる値を採用
                    total_loss = actor_loss + VALUE_LOSS_WEIGHT * critic_loss + ENTROPY_WEIGHT * entropy_loss
                    
                # 勾配を計算し、単一のオプティマイザでモデル全体の変数を更新
                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                #print('actor_loss=',actor_loss.numpy(),'critic_loss=',critic_loss.numpy(),'entropy_loss=',entropy_loss.numpy())
                avg_loss += total_loss
                avg_entropy += -entropy_loss
                num_batches += 1

        avg_loss /= num_batches
        avg_entropy /= num_batches

        # === 4. 進捗の評価と表示 (変更なし) ===
        eval_env = gym.make(ENV_NAME)
        eval_scores = []
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
            eval_scores.append(score)
        avg_evl_score = np.mean(eval_scores)
        last_episode_scores = last_episode_scores[-10:]
        avg_score = np.mean(last_episode_scores)
        
        print(f"Ep:{episode_count}, St:{total_step}, AvgScore:{avg_score:.1f}, AvgLoss:{avg_loss:.3f}, AvgEntropy:{avg_entropy:.4f}, AvgEvalScore:{avg_evl_score:.1f}")

        if avg_score >= TARGET_SCORE:
            print(f"\n目標スコア {TARGET_SCORE} を達成しました！学習を終了します。")
            break

    print("\n--- テスト実行 ---")
    env_render = gym.make("Pendulum-v1", render_mode="rgb_array")
    frames = []
    for i in range(5):
        state, _ = env_render.reset()
        done, truncated = False, False
        test_reward = 0
        
        while not (done or truncated):
            frames.append(env_render.render())
            action = get_best_action(model, state)
            state, reward, done, truncated, _ = env_render.step(action)
            test_reward += reward
        print(f"Test Episode {i+1}, Total Reward: {test_reward:.2f}")
    
    env_render.close()
    
    gif_path = 'pendulum-a2c-dlr3-nstep3.gif'
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"GIFを'{gif_path}'に保存しました。")



if __name__ == '__main__':
    main()