import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import time
import imageio

# tf.compat.v1.enable_eager_execution() # TensorFlow 2.xでは不要

# === モデルの定義 (探索安定化版) ===
def create_a2c_model(input_shape, action_dim):
    """Actor-Criticモデルを生成する (探索安定化版)"""
    input_layer = keras.layers.Input(shape=input_shape)
    common_layer = keras.layers.Dense(256, activation="relu")(input_layer)
    common_layer = keras.layers.Dense(256, activation="relu")(common_layer)

    last_init = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
    actor_mu = keras.layers.Dense(action_dim, activation="tanh", name="actor_mu", kernel_initializer=last_init)(common_layer)

    # ★★★ 改善点 1: log_stdの出力範囲を制限する ★★★
    # これにより、標準偏差が0に近づきすぎるのを防ぎ、学習を安定させる
    LOG_STD_MIN = -5.0  # log_stdの下限値
    LOG_STD_MAX = 2.0   # log_stdの上限値
    
    # log_stdを直接出力する層
    log_std_output = keras.layers.Dense(
        action_dim, 
        name="log_std_raw",
        bias_initializer=tf.keras.initializers.Constant(0.0)
        )(common_layer)
    
    # Lambdaレイヤーを使って、出力を[-5, 2]の範囲にクリッピングする
    actor_log_std = keras.layers.Lambda(
        lambda x: tf.clip_by_value(x, LOG_STD_MIN, LOG_STD_MAX), 
        name="actor_log_std"
    )(log_std_output)
    
    critic_value = keras.layers.Dense(1, activation="linear", name="critic_value")(common_layer)

    # モデルの出力にクリップ後のactor_log_stdを使用する
    model = keras.Model(inputs=input_layer, outputs=[actor_mu, actor_log_std, critic_value])
    return model

# === 行動選択関数 (TFP不使用版に書き換え) ===
def get_action(model, state, action_bound):
    """正規分布から行動をサンプリングする (TFP不使用)"""
    mu_normalized, log_std, _ = model(state.reshape((1, -1)))
    
    std = tf.exp(log_std)
    
    # tf.random.normal を使って正規分布からサンプリング
    # action = mu + noise * std と同等
    action_normalized = tf.random.normal(shape=tf.shape(mu_normalized), mean=mu_normalized, stddev=std)
    
    # サンプリングした値を環境の行動範囲にスケール
    action = action_normalized[0] * action_bound
    action = tf.clip_by_value(action, -action_bound, action_bound)
    
    return action.numpy()

def get_best_action(model, state, action_bound):
    """決定論的な行動（分布の平均値）を選択する"""
    mu_normalized, _, _ = model(state.reshape((1, -1)))
    action = mu_normalized[0] * action_bound
    return action.numpy()

# === 学習関数 (GAE導入版) ===
def update(model, optimizer, experiences, gamma, value_loss_weight, entropy_weight, standardize, action_bound, gae_lambda):
    print(experiences[-1])
    states = np.asarray([e["state"] for e in experiences], dtype=np.float32)
    actions = np.asarray([e["action"] for e in experiences], dtype=np.float32)
    rewards = np.asarray([e["reward"] for e in experiences], dtype=np.float32)
    dones = np.asarray([e["done"] for e in experiences], dtype=np.float32) # boolからfloatへ

    # 価値関数 V(s) を取得 (勾配計算には使わないのでtraining=False)
    _, _, values_tensor = model(states, training=False)
    values = values_tensor.numpy().flatten()

    # 最後の状態の価値 V(s_T) を計算
    if not experiences[-1]["done"]:
        _, _, last_v_tensor = model(experiences[-1]["n_state"].reshape((1, -1)), training=False)
        last_v = last_v_tensor.numpy()[0, 0]
    else:
        last_v = 0.0

    # ★★★ GAE (Generalized Advantage Estimation) の計算 ★★★
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0.0
    # 1ステップずらした価値のリストを作成 (最後の価値はlast_v)
    next_values = np.append(values[1:], last_v)
    
    for t in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[t]  # エピソードが終了していなければ1.0
        # 1-step TD誤差 (delta) を計算
        delta = rewards[t] + gamma * next_values[t] * next_non_terminal - values[t]
        # GAE advantageを計算
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
    
    # 割引報酬和 (returns) を計算: returns = advantages + values
    returns = advantages + values
    
    # Tensorflowのテンソルに変換
    advantages_tensor = tf.convert_to_tensor(advantages.reshape(-1, 1), dtype=tf.float32)
    returns_tensor = tf.convert_to_tensor(returns.reshape(-1, 1), dtype=tf.float32)

    with tf.GradientTape() as tape:
        mu_normalized, log_std, v_pred = model(states, training=True)
        
        # Advantageを正規化
        if standardize:
            advantages_tensor = (advantages_tensor - tf.reduce_mean(advantages_tensor)) / (tf.math.reduce_std(advantages_tensor) + 1e-8)

        # log_probとentropyの計算 (変更なし)
        std = tf.exp(log_std)
        mu_scaled = mu_normalized * action_bound
        log_prob = -0.5 * (tf.square((actions - mu_scaled) / std) + 2 * log_std + np.log(2 * np.pi))
        log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
        entropy = 0.5 + 0.5 * tf.math.log(2 * np.pi) + log_std
        entropy = tf.reduce_sum(entropy, axis=1, keepdims=True)

        # 方策損失 (ターゲットはAdvantage)
        policy_loss = -log_prob * tf.stop_gradient(advantages_tensor)

        # 価値損失 (ターゲットはReturns)
        value_loss = tf.keras.losses.huber(returns_tensor, v_pred)

        # 全体の損失
        total_loss = policy_loss + value_loss_weight * value_loss - entropy_weight * entropy
        total_loss = tf.reduce_mean(total_loss)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss.numpy(), tf.reduce_mean(entropy).numpy()


# === 学習ループ (GAE導入版) ===
def train(env, model, action_bound):
    # --- ハイパーパラメータ (GAE導入版) ---
    standardize = True
    total_timesteps = 300000 
    n_steps = 256
    gamma = 0.99
    gae_lambda = 0.95  # ★★★ GAE用の新しいハイパーパラメータ ★★★
    initial_lr = 3e-4         
    clipnorm = 0.5
    value_loss_weight = 0.5
    entropy_weight = 0.1 #0.01

    print('A2C for Pendulum-v1 (with GAE)') # 表示を更新
    print(f'standardize = {standardize}')    
    print(f'total_timesteps = {total_timesteps}')
    print(f'n_steps = {n_steps}')
    print(f'gamma = {gamma}')
    print(f'lr = {initial_lr} (Polynomial Decay, power=1.0)') # ★ 表示を更新
    print(f'clipnorm = {clipnorm}')
    print(f'value_loss_weight = {value_loss_weight}')
    print(f'entropy_weight = {entropy_weight}')
    print(f'gae_lambda = {gae_lambda}') # GAEパラメータも表示
    
    # ★ 改善点: LinearDecayの代わりにPolynomialDecayを使用
    # power=1.0に設定することで、線形減衰と全く同じ動作になります。
    lr_schedule = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_timesteps, # 全ステップを通して学習率を減衰
        end_learning_rate=1e-6,      # 最終的な学習率
        power=1.0
    )

    optimizer = Adam(learning_rate=lr_schedule, clipnorm=clipnorm)

    print("--- 学習開始 ---")
    start_time = time.time()
    all_rewards = []
    all_episode_steps = []
    all_losses = [] 
    all_entropies = []
    
    experiences = []
    episode_count = 0
    update_count = 0
    episode_reward_sum = 0
    episode_step_count = 0
    
    state, _ = env.reset()
    print('reset',state)

    for global_step in range(1, total_timesteps + 1):
        episode_step_count += 1
        
        action = get_action(model, state, action_bound)
        
        print('action',action)
        n_state, reward, done, truncated, _ = env.step(action)
        print('step',n_state)
        
        episode_reward_sum += reward

        experiences.append({
            "state": state, "action": action, "reward": reward,
            "n_state": n_state, "done": done
        })
        
        state = n_state

        if done or truncated:
            all_rewards.append(episode_reward_sum)
            all_episode_steps.append(episode_step_count)
            episode_count += 1
            episode_reward_sum = 0
            episode_step_count = 0
            state, _ = env.reset()

        # n_stepsごとに学習
        if len(experiences) >= n_steps:
            # ★★★ update関数にgae_lambdaを渡す ★★★
            loss, entropy = update(model, optimizer, experiences, gamma, value_loss_weight, entropy_weight, standardize, action_bound, gae_lambda)
            all_losses.append(loss)
            all_entropies.append(entropy)
            experiences = []
            update_count += 1


        if (global_step % (n_steps * 5) == 0) or global_step == total_timesteps:
            avg_reward = np.mean(all_rewards[-20:]) if all_rewards else -1600
            avg_episode_steps = np.mean(all_episode_steps[-20:]) if all_episode_steps else 0
            last_loss = all_losses[-1] if all_losses else 0
            last_entropy = all_entropies[-1] if all_entropies else 0
            print(f'Update:{update_count}|Step:{global_step//1000}k/{total_timesteps//1000}k|Ep:{episode_count}|AvgSteps:{avg_episode_steps:.1f}|AvgReward:{avg_reward:.1f}|Loss:{last_loss:.3f}|Entropy:{last_entropy:.3f}')
            
            if avg_reward > -200:
                print(f"環境がクリアされました！ (平均報酬: {avg_reward})")
                break
    
    print("--- 学習終了 ---")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"実行時間: {execution_time:.4f}秒")
    
    # === 結果のプロット (変更なし) ===
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(all_rewards, label='Episode Reward')
    if len(all_rewards) >= 20:
        moving_avg = np.convolve(all_rewards, np.ones(20)/20, mode='valid')
        plt.plot(np.arange(19, len(all_rewards)), moving_avg, label='Moving Average (20 ep)', color='orange')
    plt.title('Total Reward per Episode (Pendulum)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(all_losses, label='Loss', alpha=0.7)
    plt.plot(all_entropies, label='Entropy', alpha=0.7)
    plt.title('Loss and Entropy per Update')
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

    model = create_a2c_model(obs_shape, action_dim)
    model.summary()

    model_file = 'pendulum-a2c.weights.h5'
    if os.path.isfile(model_file):
        print(f"学習済みモデル {model_file} を読み込みます。")
        model.load_weights(model_file)
    else:
        train(env, model, action_bound)
        print(f"学習済みモデルを {model_file} に保存します。")
        model.save_weights(model_file)
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
            action = get_best_action(model, state, action_bound)
            state, reward, done, truncated, _ = env_render.step(action)
            test_reward += reward
        print(f"Test Episode {i+1}, Total Reward: {test_reward:.2f}")
    
    env_render.close()
    
    gif_path = 'pendulum-a2c.gif'
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"GIFを'{gif_path}'に保存しました。")