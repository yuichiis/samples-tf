import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import time
import imageio


class ActorModel(keras.Model):
    def __init__(self, input_shape, action_dim):
        super().__init__()
        self.mu_model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(action_dim, activation="tanh", name="actor_mu"),
        ])
        # logstdsを学習可能な変数として定義
        self.logstds = tf.Variable(tf.fill((action_dim,), 0.7), trainable=True, name="logstds")
        self.min_log_std = np.log(1e-3)
        self.max_log_std = np.log(50)

    @property
    def trainable_variables(self):
        return self.mu_model.trainable_variables + [self.logstds]
    
    def call(self, inputs):
        mu = self.mu_model(inputs)
        log_std = tf.clip_by_value(self.logstds, self.min_log_std, self.max_log_std)
        return mu, log_std

class CriticModel(keras.Model):
    def __init__(self, input_shape):
        super().__init__()
        self.critic_model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(1, activation="linear", name="critic_value"),
        ])

    def call(self, inputs):
        return self.critic_model(inputs)


# === モデルの定義 (パラメータ非共有・独立ネットワーク版) ===
def create_actor_critic_models(input_shape, action_dim):
    """
    ActorモデルとCriticモデルを、パラメータを共有せずに独立して生成する
    """
    # --- Actorモデルの定義 ---
    #actor_input = keras.layers.Input(shape=input_shape)
    #actor_layer = keras.layers.Dense(256, activation="relu")(actor_input)
    #actor_layer = keras.layers.Dense(256, activation="relu")(actor_layer)
    #
    ##last_init = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
    ##actor_mu = keras.layers.Dense(action_dim, activation="tanh", name="actor_mu", kernel_initializer=last_init)(actor_layer)
    #actor_mu = keras.layers.Dense(action_dim, activation="tanh", name="actor_mu")(actor_layer)
    #actor_log_std = keras.layers.Dense(action_dim, name="log_std")(actor_layer)
    #
    #MAX_LOG_STD = np.log(50)
    #MIN_LOG_STD = np.log(1e-3)
    #actor_log_std = keras.layers.Lambda(lambda x: tf.clip_by_value(x, MIN_LOG_STD, MAX_LOG_STD))(actor_log_std)    

    #MAX_LOG_STD = 2
    #MIN_LOG_STD = -5
    #scale_log_std = (MAX_LOG_STD-MIN_LOG_STD)/2
    #base_log_std = MAX_LOG_STD-scale_log_std
    #actor_log_std = keras.layers.Activation("tanh")(actor_log_std)
    #actor_log_std = keras.layers.Lambda(lambda x: x * scale_log_std + base_log_std)(actor_log_std)    
    #MAX_LOG_STD = 5
    #actor_log_std = keras.layers.Activation("tanh")(actor_log_std)
    #actor_log_std = keras.layers.Lambda(lambda x: x * MAX_LOG_STD)(actor_log_std)    

    #actor_model = keras.Model(inputs=actor_input, outputs=[actor_mu, actor_log_std], name="actor")
    
    # --- Criticモデルの定義 ---
    #critic_input = keras.layers.Input(shape=input_shape)
    #critic_layer = keras.layers.Dense(256, activation="relu")(critic_input)
    #critic_layer = keras.layers.Dense(256, activation="relu")(critic_layer)
    #
    #critic_value = keras.layers.Dense(1, activation="linear", name="critic_value")(critic_layer)
    #
    #critic_model = keras.Model(inputs=critic_input, outputs=critic_value, name="critic")

    actor_model = ActorModel(input_shape, action_dim)
    critic_model = CriticModel(input_shape)

    return actor_model, critic_model

# === 行動選択関数 ===
def get_action(actor_model, state, action_bound):
    """正規分布から行動をサンプリングする (tfp版)"""

    # 状態をバッチ次元付きのテンソルに変換
    state_tensor = tf.convert_to_tensor(state.reshape((1, -1)), dtype=tf.float32)
    
    # Actorモデルから分布のパラメータを取得
    mu_normalized, log_std = actor_model(state_tensor)
    
    # tfpを使って正規分布を構築
    std = tf.exp(log_std)
    normal_dist = tfp.distributions.Normal(loc=mu_normalized, scale=std)
    
    # 分布から行動をサンプリング
    action_normalized = normal_dist.sample() # shape: (1, action_dim)
    
    # action_boundでスケールし、環境の範囲内にクリップ
    # action_normalized[0] でバッチ次元を取り除く
    action = action_normalized[0] * action_bound
    action = tf.clip_by_value(action, -action_bound, action_bound)
    
    #action = tf.tanh(action_normalized[0]) * action_bound
    #print("{:.2f} {:.2f} {:.5f} {:.2f}".format(log_std[0].numpy(),std[0].numpy(),mu_normalized[0,0].numpy(),action[0].numpy()))
    
    return action.numpy()

def get_best_action(actor_model, state, action_bound):
    """決定論的な行動（分布の平均値）を選択する"""
    mu_normalized, _ = actor_model(state.reshape((1, -1)))
    action = mu_normalized[0] * action_bound
    return action.numpy()

# === 学習関数 (Actor/Critic分離版) ===
# === 学習関数 (2つのGradientTapeを使用する版) ===
# ★★★ 変更点 1: 戻り値を actor_loss, critic_loss, entropy に変更 ★★★
def update(actor_model, critic_model, actor_optimizer, critic_optimizer, experiences, gamma, value_loss_weight, entropy_weight, standardize, action_bound):
    states = np.asarray([e["state"] for e in experiences], dtype=np.float32)
    actions = np.asarray([e["action"] for e in experiences], dtype=np.float32)
    rewards = np.asarray([e["reward"] for e in experiences], dtype=np.float32)
    dones = np.asarray([e["done"] for e in experiences], dtype=np.bool_)

    if not dones[-1]:
        last_v = critic_model(experiences[-1]["n_state"].reshape((1, -1)), training=False)
        last_v = last_v.numpy()[0, 0]
    else:
        last_v = 0.0

    discounted_rewards = []
    G = last_v
    for r in rewards[::-1]:
        G = r + gamma * G
        discounted_rewards.append(G)
    discounted_rewards.reverse()
    discounted_rewards = np.array(discounted_rewards, dtype=np.float32).reshape(-1, 1)

    # --- Criticの更新 ---
    with tf.GradientTape() as critic_tape:
        # Criticのフォワードパス
        v = critic_model(states, training=True)
        # Huber損失を使ってCriticの損失を計算
        value_loss = tf.keras.losses.huber(discounted_rewards, v)
        critic_loss = value_loss_weight * tf.reduce_mean(value_loss)
    
    # Criticの勾配を計算して適用
    critic_grads = critic_tape.gradient(critic_loss, critic_model.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, critic_model.trainable_variables))

    # --- Actorの更新 ---
    # AdvantageはActorの損失計算に必要だが、勾配は流さない
    # vは上記critic_tapeの外で計算されたテンソルだが、値として利用可能
    advantage = discounted_rewards - v
    if standardize:
        advantage_norm = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)
    else:
        advantage_norm = advantage
    
    with tf.GradientTape() as actor_tape:
        # Actorのフォワードパス
        mu_normalized, log_std = actor_model(states, training=True)
        
        # 確率とエントロピーの計算
        std = tf.exp(log_std)
        
        normal_dist = tfp.distributions.Normal(loc=mu_normalized, scale=std)

        actions_normalized = actions / action_bound
        log_prob = tf.reduce_sum(normal_dist.log_prob(actions_normalized), axis=1, keepdims=True)
        entropy = tf.reduce_sum(normal_dist.entropy(), axis=1, keepdims=True)

        policy_loss = -log_prob * tf.stop_gradient(advantage_norm)
        actor_loss = tf.reduce_mean(policy_loss - entropy_weight * entropy)


    # Actorの勾配を計算して適用
    actor_grads = actor_tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor_model.trainable_variables))

    # ログ表示用の損失とエントロピーを返す
    return actor_loss.numpy(), critic_loss.numpy(), tf.reduce_mean(entropy).numpy()

# === 学習ループ ===
# ★★★ 変更点 2: ログとプロットのために損失を個別に保持 ★★★
def train(env, actor_model, critic_model, action_bound):
    standardize = True
    total_timesteps = 300000
    n_steps = 512
    gamma = 0.99
    actor_lr = 3e-4
    critic_lr = 1e-3
    clipnorm = 0.5
    value_loss_weight = 0.5
    entropy_weight = 0.01

    print('A2C for Pendulum-v1 (Separate Optimizers & Losses)')
    print(f'standardize = {standardize}, total_timesteps = {total_timesteps}, n_steps = {n_steps}')
    print(f'gamma = {gamma}, clipnorm = {clipnorm}, value_loss_weight = {value_loss_weight}, entropy_weight = {entropy_weight}')
    print(f'actor_lr = {actor_lr}, critic_lr = {critic_lr}')

    #actor_lr_schedule = keras.optimizers.schedules.PolynomialDecay(
    #    initial_learning_rate=actor_lr, decay_steps=total_timesteps, end_learning_rate=1e-6, power=1.0
    #)
    actor_optimizer = Adam(learning_rate=actor_lr, clipnorm=clipnorm)

    #critic_lr_schedule = keras.optimizers.schedules.PolynomialDecay(
    #    initial_learning_rate=critic_lr, decay_steps=total_timesteps, end_learning_rate=1e-6, power=1.0
    #)
    critic_optimizer = Adam(learning_rate=critic_lr, clipnorm=clipnorm)

    print("--- 学習開始 ---")
    start_time = time.time()
    all_rewards = []
    all_episode_steps = []
    # 損失を個別に保持するためのリスト
    all_actor_losses = []
    all_critic_losses = [] 
    all_entropies = []
    
    experiences = []
    episode_count = 0
    update_count = 0
    episode_reward_sum = 0
    episode_step_count = 0
    
    state, _ = env.reset()

    for global_step in range(1, total_timesteps + 1):
        episode_step_count += 1
        
        action = get_action(actor_model, state, action_bound)
        
        n_state, reward, terminated, truncated, _ = env.step(action)
        episode_reward_sum += reward

        experiences.append({
            "state": state, "action": action, "reward": reward,
            "n_state": n_state, "done": terminated or truncated
        })
        state = n_state

        if terminated or truncated:
            all_rewards.append(episode_reward_sum)
            all_episode_steps.append(episode_step_count)
            episode_count += 1
            episode_reward_sum = 0
            episode_step_count = 0
            state, _ = env.reset()

        if len(experiences) >= n_steps:
            # 個別の損失を受け取る
            actor_loss, critic_loss, entropy = update(
                actor_model, critic_model, actor_optimizer, critic_optimizer, experiences, 
                gamma, value_loss_weight, entropy_weight, standardize, action_bound
            )
            # それぞれのリストに追加
            all_actor_losses.append(actor_loss)
            all_critic_losses.append(critic_loss)
            all_entropies.append(entropy)
            experiences = []
            update_count += 1

        if (global_step % (n_steps * 5) == 0) or global_step == total_timesteps:
            avg_reward = np.mean(all_rewards[-20:]) if all_rewards else -1600
            avg_episode_steps = np.mean(all_episode_steps[-20:]) if all_episode_steps else 0
            last_actor_loss = all_actor_losses[-1] if all_actor_losses else 0
            last_critic_loss = all_critic_losses[-1] if all_critic_losses else 0
            last_entropy = all_entropies[-1] if all_entropies else 0
            # ログ表示を更新
            print(f'St:{global_step//1000}k|Ep:{episode_count}|AvgSt:{avg_episode_steps:.1f}|AvgRwd:{avg_reward:.1f}|'
                  f'ActorL:{last_actor_loss:.3f}|CriticL:{last_critic_loss:.3f}|Entr:{last_entropy:.3f}')
            
            if avg_reward > -200:
                print(f"環境がクリアされました！ (平均報酬: {avg_reward})")
                break
    
    print("--- 学習終了 ---")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"実行時間: {execution_time:.4f}秒")
    
    # ★★★ 変更点 3: プロットを更新 ★★★
    plt.figure(figsize=(14, 10))
    # 報酬プロット
    plt.subplot(2, 2, 1)
    plt.plot(all_rewards, label='Episode Reward')
    if len(all_rewards) >= 20:
        moving_avg = np.convolve(all_rewards, np.ones(20)/20, mode='valid')
        plt.plot(np.arange(19, len(all_rewards)), moving_avg, label='Moving Average (20 ep)', color='orange')
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    # Actor損失プロット
    plt.subplot(2, 2, 2)
    plt.plot(all_actor_losses, label='Actor Loss', color='blue')
    plt.title('Actor Loss per Update')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Critic損失プロット
    plt.subplot(2, 2, 3)
    plt.plot(all_critic_losses, label='Critic Loss', color='red')
    plt.title('Critic Loss per Update')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # エントロピープロット
    plt.subplot(2, 2, 4)
    plt.plot(all_entropies, label='Entropy', color='green')
    plt.title('Entropy per Update')
    plt.xlabel('Update Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# === メイン処理 (変更なし) ===
if __name__ == '__main__':
    env = gym.make("Pendulum-v1")
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    actor_model, critic_model = create_actor_critic_models(obs_shape, action_dim)
    print("--- Actor Model ---")
    actor_model.summary()
    print("\n--- Critic Model ---")
    critic_model.summary()

    actor_model_file = 'pendulum-a2c-dlr3-actor2.weights.h5'
    critic_model_file = 'pendulum-a2c-dlr3-critic2.weights.h5'
    
    if os.path.isfile(actor_model_file) and os.path.isfile(critic_model_file):
        print(f"学習済みモデル {actor_model_file}, {critic_model_file} を読み込みます。")
        actor_model.load_weights(actor_model_file)
        critic_model.load_weights(critic_model_file)
    else:
        train(env, actor_model, critic_model, action_bound)
        print(f"学習済みモデルを保存します。")
        actor_model.save_weights(actor_model_file)
        critic_model.save_weights(critic_model_file)
    env.close()

    print("\n--- テスト実行 ---")
    env_render = gym.make("Pendulum-v1", render_mode="rgb_array")
    frames = []
    for i in range(3):
        state, _ = env_render.reset()
        done, truncated = False, False
        test_reward = 0
        
        while not (done or truncated):
            frames.append(env_render.render())
            action = get_best_action(actor_model, state, action_bound)
            state, reward, done, truncated, _ = env_render.step(action)
            test_reward += reward
        print(f"Test Episode {i+1}, Total Reward: {test_reward:.2f}")
    
    env_render.close()
    
    gif_path = 'pendulum-a2c-dlr3-nstep2.gif'
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"GIFを'{gif_path}'に保存しました。")
