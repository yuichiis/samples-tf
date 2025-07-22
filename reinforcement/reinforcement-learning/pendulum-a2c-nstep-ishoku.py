import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import time
import imageio

# -------------------------------------------------------------------
# 1. 活性化関数の定義
# -------------------------------------------------------------------
@keras.utils.register_keras_serializable()
def mish(x):
    """Mish活性化関数"""
    return x * tf.math.tanh(tf.math.softplus(x))

# -------------------------------------------------------------------
# 2. Actor-Criticモデルの定義 (PyTorch成功例準拠)
# -------------------------------------------------------------------
class Actor(keras.Model):
    """ Actorモデル: 行動の平均値(mean)と標準偏差(std)を計算 """
    def __init__(self, action_dim):
        super().__init__()
        # Actorネットワーク本体
        self.model = keras.Sequential([
            keras.layers.Dense(64, activation=mish),
            keras.layers.Dense(64, activation=mish),
            keras.layers.Dense(action_dim, activation='linear')
        ])
        
        # 状態に依存しない、学習可能な対数標準偏差
        self.log_stds = tf.Variable(tf.zeros(action_dim), trainable=True, name='log_stds')

    def call(self, states):
        """ 状態(states)を入力とし、行動の平均(means)と標準偏差(stds)を出力 """
        means = self.model(states)
        # 標準偏差が極端な値にならないようにクリッピング
        stds = tf.clip_by_value(tf.exp(self.log_stds), 1e-3, 50)
        return means, stds

class Critic(keras.Model):
    """ Criticモデル: 状態価値(V)を計算 """
    def __init__(self):
        super().__init__()
        self.model = keras.Sequential([
            keras.layers.Dense(64, activation=mish),
            keras.layers.Dense(64, activation=mish),
            keras.layers.Dense(1, activation='linear')
        ])

    def call(self, states):
        """ 状態(states)を入力とし、状態価値(value)を出力 """
        return self.model(states)

# -------------------------------------------------------------------
# 3. 行動選択関数 (形状バグ修正済み)
# -------------------------------------------------------------------
def get_action(actor_model, state, action_bound):
    """ 学習中の行動選択: 正規分布からサンプリング """
    means, stds = actor_model(state.reshape((1, -1)))
    # 平均と標準偏差を持つ正規分布からアクションをサンプリング
    action = tf.random.normal(shape=tf.shape(means), mean=means, stddev=stds)
    # アクションを環境の有効範囲にクリッピング
    action = tf.clip_by_value(action, -action_bound, action_bound)
    # 正しいスカラー値を返すためのインデックス指定
    return action.numpy()[0, 0]

def get_best_action(actor_model, state, action_bound):
    """ テスト時の行動選択: 決定論的に平均値を選択 """
    means, _ = actor_model(state.reshape((1, -1)))
    action = tf.clip_by_value(means, -action_bound, action_bound)
    # 正しいスカラー値を返すためのインデックス指定
    return action.numpy()[0, 0]

# -------------------------------------------------------------------
# 4. 学習関数 (報酬正規化、形状バグ修正済み)
# -------------------------------------------------------------------
def update(actor, critic, actor_optim, critic_optim, experiences, gamma, entropy_beta, max_grad_norm):
    """ 経験バッファを使ってActorとCriticを更新する """
    states = np.asarray([e["state"] for e in experiences], dtype=np.float32)
    actions = np.asarray([e["action"] for e in experiences], dtype=np.float32).reshape(-1, 1)
    rewards = np.asarray([e["reward"] for e in experiences], dtype=np.float32)
    next_states = np.asarray([e["n_state"] for e in experiences], dtype=np.float32)
    dones = np.asarray([e["done"] for e in experiences], dtype=np.float32)

    # 報酬を正規化して学習を安定させる (非常に重要)
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

    # 1ステップTDターゲットを計算
    next_values = critic(next_states, training=False)
    td_target = rewards.reshape(-1, 1) + gamma * next_values * (1 - dones.reshape(-1, 1))
    
    # --- Criticの学習 ---
    with tf.GradientTape() as tape:
        values = critic(states, training=True)
        # CriticはTDターゲットに近づくように学習
        critic_loss = tf.reduce_mean(tf.square(td_target - values))
        
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    critic_grads, _ = tf.clip_by_global_norm(critic_grads, max_grad_norm)
    critic_optim.apply_gradients(zip(critic_grads, critic.trainable_variables))

    # --- Actorの学習 ---
    with tf.GradientTape() as tape:
        # Advantage (TDターゲット - 現在の価値)
        advantage = td_target - values
        
        means, stds = actor(states, training=True)
        
        # 対数確率密度(log_prob)の計算
        log_prob = -0.5 * (tf.square((actions - means) / stds) + 2 * tf.math.log(stds) + np.log(2 * np.pi))
        
        # エントロピーの計算
        entropy = 0.5 + 0.5 * tf.math.log(2 * np.pi) + tf.math.log(stds)
        
        # Actorの損失関数
        actor_loss = -tf.reduce_mean(log_prob * tf.stop_gradient(advantage)) - tf.reduce_mean(entropy) * entropy_beta

    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    actor_grads, _ = tf.clip_by_global_norm(actor_grads, max_grad_norm)
    actor_optim.apply_gradients(zip(actor_grads, actor.trainable_variables))
    
    return actor_loss.numpy(), critic_loss.numpy(), tf.reduce_mean(entropy).numpy()

# -------------------------------------------------------------------
# 5. 学習ループ (詳細ログ出力)
# -------------------------------------------------------------------
def train(env, actor, critic, action_bound):
    # --- ハイパーパラメータ (PyTorch成功例準拠) ---
    total_episodes = 500
    episode_length = 200 
    n_steps_on_memory = 16 # このステップ数ごとに学習
    gamma = 0.99
    actor_lr = 4e-5#4e-4
    critic_lr = 4e-3
    entropy_beta = 1e-4
    max_grad_norm = 0.5

    print('total_episodes:',total_episodes)
    print('episode_length:',episode_length)
    print('n_steps_on_memory:',n_steps_on_memory)
    print('gamma:',gamma)
    print('actor_lr:',actor_lr)
    print('critic_lr:',critic_lr)
    print('entropy_beta:',entropy_beta)
    print('max_grad_norm:',max_grad_norm)
    
    # ActorとCriticで別々のオプティマイザを用意
    actor_optim = Adam(learning_rate=actor_lr)
    critic_optim = Adam(learning_rate=critic_lr)

    print("--- 学習開始 (最終修正版) ---")
    
    # ログ収集用リスト
    all_rewards = []
    all_actor_losses = []
    all_critic_losses = []
    all_entropies = []
    
    state, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    update_count = 0
    
    total_steps = (episode_length * total_episodes)
    for global_step in range(1, total_steps + 1):
        
        experiences = []
        for _ in range(n_steps_on_memory):
            action = get_action(actor, state, action_bound)
            # actionはスカラーなので、リストで囲んで渡す
            n_state, reward, terminated, truncated, _ = env.step([action])
            done = terminated or truncated
            episode_reward += reward
            
            experiences.append({
                "state": state, "action": action, "reward": reward,
                "n_state": n_state, "done": done
            })
            
            state = n_state
            if done:
                all_rewards.append(episode_reward)
                episode_count += 1
                
                # 10エピソードごとに詳細ログを出力
                if episode_count % 10 == 0:
                    avg_reward = np.mean(all_rewards[-10:]) if all_rewards else 0
                    last_actor_loss = all_actor_losses[-1] if all_actor_losses else 0
                    last_critic_loss = all_critic_losses[-1] if all_critic_losses else 0
                    last_entropy = all_entropies[-1] if all_entropies else 0
                    print(f'Update:{update_count}|Ep:{episode_count}|AvgReward:{avg_reward:.1f}|'
                          f'ActorLoss:{last_actor_loss:.3f}|CriticLoss:{last_critic_loss:.3f}|'
                          f'Entropy:{last_entropy:.3f}')
                
                episode_reward = 0
                state, _ = env.reset()
                
                if episode_count >= total_episodes:
                    break
        
        if episode_count >= total_episodes:
            break

        actor_loss, critic_loss, entropy = update(actor, critic, actor_optim, critic_optim, experiences, gamma, entropy_beta, max_grad_norm)
        all_actor_losses.append(actor_loss)
        all_critic_losses.append(critic_loss)
        all_entropies.append(entropy)
        update_count += 1

    print("--- 学習終了 ---")
    
    # --- 結果のプロット ---
    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards, label='Episode Reward')
    if len(all_rewards) >= 10:
        moving_avg = np.convolve(all_rewards, np.ones(10)/10, mode='valid')
        plt.plot(np.arange(9, len(all_rewards)), moving_avg, label='Moving Average (10 ep)', color='orange')
    plt.title('Total Reward per Episode (Pendulum)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------------------------------------------------
# 6. メイン処理
# -------------------------------------------------------------------
if __name__ == '__main__':
    env = gym.make("Pendulum-v1")
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    actor = Actor(action_dim)
    critic = Critic()
    
    # ダミー入力でモデルをビルドしてサマリーを表示
    actor(np.random.rand(1, obs_shape[0]))
    critic(np.random.rand(1, obs_shape[0]))
    actor.summary()
    critic.summary()

    train(env, actor, critic, action_bound)
    
    # --- テスト実行 ---
    print("\n--- テスト実行 ---")
    env_render = gym.make("Pendulum-v1", render_mode="rgb_array")
    frames = []
    for i in range(3):
        state, _ = env_render.reset()
        done, truncated = False, False
        test_reward = 0
        
        while not (done or truncated):
            frames.append(env_render.render())
            action = get_best_action(actor, state, action_bound)
            # actionはスカラーなので、リストで囲んで渡す
            state, reward, done, truncated, _ = env_render.step([action])
            test_reward += reward
        print(f"Test Episode {i+1}, Total Reward: {test_reward:.2f}")
    
    env_render.close()
    
    gif_path = 'pendulum-a2c-final.gif'
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"GIFを'{gif_path}'に保存しました。")