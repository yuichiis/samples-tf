import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import time
import imageio

# === モデルの定義 ===
def create_a2c_model(input_shape, num_actions):
    """Actor-Criticモデルを生成する"""
    input_layer = keras.layers.Input(shape=input_shape)
    common_layer = keras.layers.Dense(256, activation="relu")(input_layer)
    common_layer = keras.layers.Dense(256, activation="relu")(common_layer)
    actor_logits = keras.layers.Dense(num_actions, activation="linear", name="actor_logits")(common_layer)
    critic_value = keras.layers.Dense(1, activation="linear", name="critic_value")(common_layer)
    model = keras.Model(inputs=input_layer, outputs=[actor_logits, critic_value])
    return model

# === 行動選択関数 ===
def get_action(model, state):
    logits, _ = model(state.reshape((1, -1)))
    probs = tf.nn.softmax(logits)
    return np.random.choice(len(probs[0]), p=probs[0].numpy())

def get_best_action(model, state):
    logits, _ = model(state.reshape((1, -1)))
    return tf.argmax(logits[0]).numpy()

# === 学習関数 (再修正版) ===
def update(model, optimizer, experiences, gamma, value_loss_weight, entropy_weight, standardize):
    states = np.asarray([e["state"] for e in experiences], dtype=np.float32)
    actions = np.asarray([e["action"] for e in experiences], dtype=np.int32)
    rewards = np.asarray([e["reward"] for e in experiences], dtype=np.float32)
    # n_stepsの最後のdoneフラグだけでなく、途中のものも必要
    dones = np.asarray([e["done"] for e in experiences], dtype=np.bool_)

    # 最後の状態の価値を計算（ブートストラップ）
    # n_stepsの最後がエピソードの終わりでなければ、その先の価値を見積もる
    if not dones[-1]:
        last_state = experiences[-1]["n_state"].reshape((1, -1))
        _, last_v = model(last_state)
        last_v = last_v.numpy()[0, 0]
    else:
        last_v = 0.0

    # === ★★★ ここからが重要な修正点 ★★★ ===
    # 割引報酬和 (G_t) を後ろから計算する
    discounted_rewards = []
    G = last_v
    # experiencesを逆順にループ
    for r, done in zip(rewards[::-1], dones[::-1]):
        # もしエピソードが終了した時点なら、割引計算をリセット
        if done:
            G = 0
        # G_t = r_t + gamma * G_{t+1} を計算
        G = r + gamma * G
        discounted_rewards.append(G)
    
    # リストを正しい順序に戻す
    discounted_rewards.reverse()
    discounted_rewards = np.array(discounted_rewards, dtype=np.float32).reshape(-1, 1)
    # === ★★★ 修正ここまで ★★★ ===

    onehot_actions = tf.one_hot(actions, model.output[0].shape[1])

    with tf.GradientTape() as tape:
        logits, v = model(states, training=True)

        advantage = discounted_rewards - tf.stop_gradient(v)
        
        if standardize:
            advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)

        action_probs = tf.nn.softmax(logits)
        selected_action_probs = tf.reduce_sum(action_probs * onehot_actions, axis=1, keepdims=True)
        selected_action_probs = tf.clip_by_value(selected_action_probs, 1e-10, 1.0)
        
        policy_loss = -tf.math.log(selected_action_probs) * advantage

        value_loss = tf.keras.losses.huber(discounted_rewards, v)

        action_probs_clipped = tf.clip_by_value(action_probs, 1e-10, 1.0)
        entropy = -tf.reduce_sum(action_probs_clipped * tf.math.log(action_probs_clipped), axis=1, keepdims=True)
        
        # Total Loss: policy lossとvalue lossは最小化、entropyは最大化(-entropyを最小化)
        total_loss = policy_loss + value_loss_weight * value_loss - entropy_weight * entropy
        total_loss = tf.reduce_mean(total_loss)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # 報告用のエントロピーは正の値にする
    return total_loss.numpy(), tf.reduce_mean(-entropy).numpy()

def train(env,model):
    # --- ハイパーパラメータ (修正済み) ---
    standardize = True
    total_timesteps = 750000 #500000 #250000
    n_steps = 256 #128
    gamma = 0.99
    # lr = 3e-4 #7e-4
    clipnorm = 0.5
    value_loss_weight = 0.5
    entropy_weight = 0.01
    # 学習率スケジューラの設定
    initial_learning_rate = 3e-4  # 開始学習率 (現在の値)
    end_learning_rate = 1e-5      # 終了学習率 (非常に小さい値)

    print('A2C 通常版')
    print('standardize =',standardize)    
    print('total_timesteps =',total_timesteps)
    print('n_steps =',n_steps)
    print('gamma =',gamma)
    print('initial_learning_rate =',initial_learning_rate)
    print('end_learning_rate =',end_learning_rate)
    #print('lr =',lr)
    print('clipnorm =',clipnorm)
    print('value_loss_weight =',value_loss_weight)
    print('entropy_weight =',entropy_weight)

    # total_timestepsで学習率がend_learning_rateに到達するように設定
    decay_steps = total_timesteps 

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate,
        decay_steps,
        end_learning_rate,
        power=1.0) # power=1.0で線形減衰

    # オプティマイザに学習率の代わりにスケジューラを渡す
    optimizer = Adam(learning_rate=lr_schedule, clipnorm=0.5)
    # ↑↑↑ ここまで追加 ↑↑↑

    # optimizer = Adam(learning_rate=lr, clipnorm=clipnorm) # <-- この行を置き換える

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

    for global_step in range(1, total_timesteps + 1):
        episode_step_count += 1
        action = get_action(model, state)
        n_state, reward, done, truncated, _ = env.step(action)
        
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

        if len(experiences) >= n_steps:
            loss, entropy = update(model, optimizer, experiences, gamma, value_loss_weight, entropy_weight, standardize)
            all_losses.append(loss)
            all_entropies.append(entropy)
            experiences = []
            update_count += 1

        if (global_step % (n_steps*10) == 0) or global_step == total_timesteps:
            avg_reward = np.mean(all_rewards[-20:])
            avg_episode_steps = np.mean(all_episode_steps[-20:])
            last_loss = all_losses[-1] if all_losses else 0
            last_entropy = all_entropies[-1] if all_entropies else 0
            print(f'Update:{update_count}|Step:{global_step}/{total_timesteps//1000}k|Ep:{episode_count}|AvgSteps:{avg_episode_steps:.1f}|AvgReward:{avg_reward:.1f}|Loss:{last_loss:.3f}|Entropy:{last_entropy:.3f}')
            if avg_reward > 475:
                print(f"環境がクリアされました！ (平均報酬: {avg_reward})")
                break
    
    print("--- 学習終了 ---")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"実行時間: {execution_time:.4f}秒")
    
    # ... (プロットとテストのコードは変更なし) ...
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(all_rewards, label='Episode Reward')
    if len(all_rewards) >= 20:
        moving_avg = np.convolve(all_rewards, np.ones(20)/20, mode='valid')
        plt.plot(np.arange(19, len(all_rewards)), moving_avg, label='Moving Average (20 ep)', color='orange')
    plt.title('Total Reward per Episode')
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


# === メイン処理 ===
if __name__ == '__main__':
    env = gym.make("LunarLander-v3")
    obs_shape = env.observation_space.shape
    nb_actions = env.action_space.n
    model = create_a2c_model(obs_shape, nb_actions)
    model.summary()

    model_file = 'lunarlander-a2c-nstep.weights.h5'
    if os.path.isfile(model_file):
        model.load_weights(model_file)
    else:
        train(env,model)
        # 9. 学習済みモデルの保存と評価
        model.save(model_file)
    env.close()

    print("\n--- テスト実行 ---")
    env_render = gym.make("LunarLander-v3", render_mode="rgb_array")
    for i in range(1):
        state, _ = env_render.reset()
        done, truncated = False, False
        test_reward = 0
        test_steps = 0
        frames = []
        while not (done or truncated):
            test_steps += 1
            frames.append(env_render.render())
            action = get_best_action(model, state)
            state, reward, done, truncated, _ = env_render.step(action)
            test_reward += reward
        print(f"Test Episode {i+1}, Test Steps: {test_steps}, Total Reward: {test_reward}")
    env_render.close()
    imageio.mimsave('lunarlander-a2c-nstep.gif', frames, fps=30)
    print(f"GIFを'lunarlander-a2c-nstep.gif'に保存しました。最終報酬: {test_reward:.2f}")

