import numpy as np
import tensorflow as tf
from tensorflow import keras
# ### RL-ZOO設定 ### AdamからRMSpropに変更
from tensorflow.keras.optimizers import RMSprop
import tensorflow_probability as tfp
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import time
import imageio



# このクラスをコードの冒頭に追加します
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        """
        Welford's online algorithmを使用して移動平均と分散を計算します。
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        """
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean, batch_var, batch_count = np.mean(x, axis=0), np.var(x, axis=0), x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

# このクラスもコードの冒頭に追加します
class NormalizeWrapper(gym.Wrapper):
    def __init__(self, env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99):
        super().__init__(env)
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = 1e-8

        # 観測と報酬の正規化用
        if self.norm_obs:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        if self.norm_reward:
            # 報酬はスカラーなのでshapeは空タプル
            self.ret_rms = RunningMeanStd(shape=())
        
        self.returns = np.zeros(())

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # ### 修正点 1 ### 元の報酬を保存しておく
        unnormalized_reward = reward

        # 報酬の正規化
        if self.norm_reward:
            # 割引リターンを更新
            self.returns = self.returns * self.gamma + reward
            self.ret_rms.update(np.array([self.returns]))
            # 報酬をスケーリング
            reward = reward / (np.sqrt(self.ret_rms.var) + self.epsilon)
            reward = np.clip(reward, -self.clip_reward, self.clip_reward)

        # 観測の正規化
        if self.norm_obs:
            self.obs_rms.update(np.array([obs]))
            obs = (obs - self.obs_rms.mean) / (np.sqrt(self.obs_rms.var) + self.epsilon)
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)

        # エピソードが終了したらリターンをリセット
        if terminated or truncated:
            self.returns = 0.0

        # ### 修正点 2 ### info辞書に元の報酬を追加
        info['unnormalized_reward'] = unnormalized_reward

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # 観測の正規化
        if self.norm_obs:
            self.obs_rms.update(np.array([obs]))
            obs = (obs - self.obs_rms.mean) / (np.sqrt(self.obs_rms.var) + self.epsilon)
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        
        # リターンをリセット
        self.returns = 0.0
        
        return obs, info

class ActorModel(keras.Model):
    def __init__(self, input_shape, action_dim):
        super().__init__()
        # ### RL-ZOO設定 ### ネットワーク構造を変更 (256,relu -> 64,tanh)
        self.common_net = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Dense(64, activation="tanh"),
            keras.layers.Dense(64, activation="tanh"),
        ])

        # muヘッドは変更なし
        last_init = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
        self.mu_head = keras.layers.Dense(
            action_dim, activation="tanh", name="actor_mu", 
            kernel_initializer=last_init
        )

        # ### RL-ZOO設定 ### log_stdの初期値を-2.0に設定
        log_std_init = 'zeros'
        bias_init = tf.keras.initializers.Constant(-2.0)
        self.log_std_head = keras.layers.Dense(
            action_dim, name="log_std",
            kernel_initializer=log_std_init,
            bias_initializer=bias_init
        )

        # log_stdのクリッピング範囲は元のままでも良いですが、参考までに設定
        self.min_log_std = -5.0
        self.max_log_std = 2.0

    def call(self, inputs):
        features = self.common_net(inputs)
        mu = self.mu_head(features)
        log_std = self.log_std_head(features)
        log_std = tf.clip_by_value(log_std, self.min_log_std, self.max_log_std)
        return mu, log_std

class CriticModel(keras.Model):
    def __init__(self, input_shape):
        super().__init__()
        # ### RL-ZOO設定 ### ネットワーク構造を変更 (256,relu -> 64,tanh)
        self.critic_model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Dense(64, activation="tanh"),
            keras.layers.Dense(64, activation="tanh"),
            keras.layers.Dense(1, activation="linear", name="critic_value"),
        ])

    def call(self, inputs):
        return self.critic_model(inputs)

# create_actor_critic_modelsは変更の必要がないため、元のコードから削除し、
# ActorModel/CriticModelを直接呼び出すようにします。

# === 行動選択関数 (変更なし) ===
def get_action(actor_model, state, action_bound):
    mu_normalized, log_std = actor_model(state.reshape((1, -1)))
    scaled_mu = mu_normalized * action_bound
    dist = tfp.distributions.Normal(loc=scaled_mu, scale=tf.math.exp(log_std))
    action_normalized = dist.sample()
    action = tf.clip_by_value(action_normalized[0], -action_bound, action_bound)
    return action.numpy()

def get_best_action(actor_model, state, action_bound):
    mu_normalized, _ = actor_model(state.reshape((1, -1)))
    action = mu_normalized[0] * action_bound
    return action.numpy()

# ### RL-ZOO設定 ### GAEを実装するためにupdate関数を大幅に修正
def update(actor_model, critic_model, actor_optimizer, critic_optimizer, experiences, 
           gamma, gae_lambda, value_loss_weight, entropy_weight, standardize, action_bound):
    
    states = np.asarray([e["state"] for e in experiences], dtype=np.float32)
    actions = np.asarray([e["action"] for e in experiences], dtype=np.float32)
    rewards = np.asarray([e["reward"] for e in experiences], dtype=np.float32)
    dones = np.asarray([e["done"] for e in experiences], dtype=np.bool_)
    n_states = np.asarray([e["n_state"] for e in experiences], dtype=np.float32)

    # V(s_t) と V(s_{t+1}) を一括で計算
    # 勾配計算には使わないので training=False
    values = critic_model(states, training=False).numpy().flatten()
    next_values = critic_model(n_states, training=False).numpy().flatten()
    
    # GAE (Generalized Advantage Estimation) の計算
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae_lam = 0
    for t in reversed(range(len(rewards))):
        # 経験の最後のエピソードが完了していなければ、最後のV(s_{t+1})を使う
        # そうでなければ、V(s_{t+1})は0として扱う
        if dones[t]:
            next_non_terminal = 0
        else:
            next_non_terminal = 1.0

        delta = rewards[t] + gamma * next_values[t] * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * last_gae_lam * next_non_terminal
    
    # Returns (ターゲット価値) = Advantages + V(s_t)
    returns = (advantages + values).reshape(-1, 1)

    # Advantageの正規化 (RL Zooではデフォルトで有効)
    if standardize:
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    
    advantages = advantages.reshape(-1, 1)

    # --- Criticの更新 ---
    with tf.GradientTape() as critic_tape:
        v = critic_model(states, training=True)
        value_loss = tf.keras.losses.huber(returns, v)
        critic_loss = value_loss_weight * tf.reduce_mean(value_loss)
    
    critic_grads = critic_tape.gradient(critic_loss, critic_model.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, critic_model.trainable_variables))

    # --- Actorの更新 ---
    with tf.GradientTape() as actor_tape:
        mu_normalized, log_std = actor_model(states, training=True)
        
        mu_scaled = mu_normalized * action_bound
        dist = tfp.distributions.Normal(loc=mu_scaled, scale=tf.math.exp(log_std))
        entropy = dist.entropy()
        log_prob = dist.log_prob(actions)

        policy_loss = -log_prob * tf.stop_gradient(advantages)
        
        actor_loss = tf.reduce_mean(policy_loss - entropy_weight * entropy)

    actor_grads = actor_tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor_model.trainable_variables))

    return actor_loss.numpy(), critic_loss.numpy(), tf.reduce_mean(entropy).numpy()

# === 学習ループ ===
# === 学習ループ (PyTorch版) ★★★最終修正案★★★ ===
def train(env, actor_critic, action_bound, device):
    standardize = False
    total_timesteps = 1_000_000
    
    # --- ここからが重要な変更 ---
    # RL Zooのパラメータ
    n_envs_zoo = 8  # RL Zooで使われている並列環境数
    n_steps_zoo = 8   # RL Zooのn_steps
    
    # シングル環境でRL Zooのバッチサイズを再現するための更新間隔
    # 1回の更新に使うデータのサイズを n_envs * n_steps に合わせる
    update_interval = n_envs_zoo * n_steps_zoo  # 8 * 8 = 64
    
    # --- ここまでが重要な変更 ---

    gamma = 0.9
    gae_lambda = 0.9
    
    lr_initial = 7e-4
    lr_final = 1e-6
    
    max_grad_norm = 0.5
    value_loss_weight = 0.5
    entropy_weight = 0.0

    print('A2C for Pendulum-v1 (PyTorch - Single Env Emulation)')
    print(f'standardize = {standardize}, total_timesteps = {total_timesteps}')
    print(f'Simulating {n_envs_zoo} envs with update_interval = {update_interval}')

    optimizer = RMSprop(actor_critic.parameters(), lr=lr_initial, alpha=0.99, eps=1e-5)

    print("--- 学習開始 ---")
    start_time = time.time()
    all_rewards, all_episode_steps = [], []
    all_actor_losses, all_critic_losses, all_entropies = [], [], []
    
    experiences = []
    episode_count, episode_reward_sum, episode_step_count = 0, 0, 0
    state, _ = env.reset()

    for global_step in range(1, total_timesteps + 1):
        # 学習率の線形減衰 (変更なし)
        frac = 1.0 - (global_step - 1.0) / total_timesteps
        lr_now = frac * (lr_initial - lr_final) + lr_final
        optimizer.param_groups[0]['lr'] = lr_now

        episode_step_count += 1
        action = get_action(actor_critic, state, action_bound, device)
        clipped_action = np.clip(action, -action_bound, action_bound)
        
        n_state, reward, terminated, truncated, info = env.step(clipped_action)
        episode_reward_sum += info['unnormalized_reward']

        experiences.append({
            "state": state, "action": clipped_action, "reward": reward,
            "n_state": n_state, "done": terminated or truncated
        })
        state = n_state

        if terminated or truncated:
            all_rewards.append(episode_reward_sum)
            all_episode_steps.append(episode_step_count)
            episode_count += 1
            episode_reward_sum, episode_step_count = 0, 0
            state, _ = env.reset()

        # ★★★ ここが一番の変更点 ★★★
        # 64ステップ分のデータが溜まったら学習する
        if len(experiences) >= update_interval:
            actor_loss, critic_loss, entropy = update(
                actor_critic, optimizer, experiences, gamma, gae_lambda, value_loss_weight,
                entropy_weight, max_grad_norm, standardize, action_bound, device
            )
            all_actor_losses.append(actor_loss)
            all_critic_losses.append(critic_loss)
            all_entropies.append(entropy)
            experiences = [] # 経験バッファをクリア

        # ログ表示の頻度を調整
        if (global_step % (update_interval * 30) == 0) or global_step == total_timesteps:
            avg_reward = np.mean(all_rewards[-20:]) if all_rewards else -1600
            print(f'St:{global_step//1000}k | Ep:{episode_count} | AvgRwd:{avg_reward:.1f} | '
                  f'ActorL:{np.mean(all_actor_losses[-30:]):.3f} | CriticL:{np.mean(all_critic_losses[-30:]):.3f} | '
                  f'Entr:{np.mean(all_entropies[-30:]):.3f} | LR:{lr_now:.2e}')
            
            if avg_reward > -200 and len(all_rewards) > 20:
                print(f"環境がクリアされました！ (平均報酬: {avg_reward})")
                break
    
    print("--- 学習終了 ---")
    print(f"実行時間: {time.time() - start_time:.4f}秒")

    # プロット部分は変更なしのため省略
    # ... (元のコードのプロット部分をここに挿入) ...
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


# === メイン処理 ===
if __name__ == '__main__':
    env_raw = gym.make("Pendulum-v1")
    # ### NORMALIZATION ### ラッパーで環境を包む
    # RL Zooの設定に合わせてgammaを0.9に設定
    env = NormalizeWrapper(env_raw, gamma=0.9) 

    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    # モデルのインスタンス化
    actor_model = ActorModel(obs_shape, action_dim)
    critic_model = CriticModel(obs_shape)

    # モデルのビルド（summary表示のため）
    actor_model(tf.zeros(shape=(1,)+obs_shape))
    critic_model(tf.zeros(shape=(1,)+obs_shape))
    
    print("--- Actor Model (RL Zoo Config) ---")
    actor_model.summary()
    print("\n--- Critic Model (RL Zoo Config) ---")
    critic_model.summary()

    # ファイル名は変えておくと良いでしょう
    actor_model_file = 'pendulum-a2c-rlzoo-norm-actor.weights.h5'
    critic_model_file = 'pendulum-a2c-rlzoo-norm-critic.weights.h5'
    
    if os.path.isfile(actor_model_file) and os.path.isfile(critic_model_file):
        print(f"学習済みモデル {actor_model_file}, {critic_model_file} を読み込みます。")
        actor_model.load_weights(actor_model_file)
        critic_model.load_weights(critic_model_file)
        # TODO: 学習済みの正規化パラメータもロードする必要がある
    else:
        train(env, actor_model, critic_model, action_bound)
        print(f"学習済みモデルを保存します。")
        actor_model.save_weights(actor_model_file)
        critic_model.save_weights(critic_model_file)
        # TODO: 学習した正規化パラメータも保存する必要がある
    env.close()

    print("\n--- テスト実行 ---")
    env_render = gym.make("Pendulum-v1", render_mode="rgb_array")
    frames = []

    # 学習済みの正規化パラメータを取得
    obs_mean = env.obs_rms.mean
    obs_var = env.obs_rms.var
    epsilon = 1e-8
    clip_obs = 10.0

    for i in range(3): # 3エピソードに短縮
        state, _ = env_render.reset()
        done, truncated = False, False
        test_reward = 0
        
        while not (done or truncated):
            frames.append(env_render.render())

            # ### NORMALIZATION ### テスト時も観測を正規化する
            normalized_state = (state - obs_mean) / (np.sqrt(obs_var) + epsilon)
            normalized_state = np.clip(normalized_state, -clip_obs, clip_obs)

            action = get_best_action(actor_model, normalized_state, action_bound)
            state, reward, done, truncated, _ = env_render.step(action)
            test_reward += reward
        print(f"Test Episode {i+1}, Total Reward: {test_reward:.2f}")
    
    env_render.close()
    
    gif_path = 'pendulum-a2c-rlzoo-norm.gif'
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"GIFを'{gif_path}'に保存しました。")
    