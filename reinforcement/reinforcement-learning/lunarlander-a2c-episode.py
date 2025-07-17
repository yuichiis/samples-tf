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

# === 学習関数 (修正・改善版) ===
def update(model, optimizer, experiences, gamma, value_loss_weight, entropy_weight, standardize):
    states = np.asarray([e["state"] for e in experiences], dtype=np.float32)
    actions = np.asarray([e["action"] for e in experiences], dtype=np.int32)
    rewards = np.asarray([e["reward"] for e in experiences], dtype=np.float32)

    # === 割引報酬和 (G_t) の計算 (エピソード更新用に簡略化) ===
    # エピソード単位の更新なので、最後の状態の価値は常に0から計算を開始する
    discounted_rewards = []
    G = 0.0
    # experiencesを逆順にループし、G_t = r_t + gamma * G_{t+1} を計算
    for r in rewards[::-1]:
        G = r + gamma * G
        discounted_rewards.append(G)
    
    # リストを正しい順序に戻す
    discounted_rewards.reverse()
    discounted_rewards = np.array(discounted_rewards, dtype=np.float32).reshape(-1, 1)
    # === 修正ここまで ===

    onehot_actions = tf.one_hot(actions, model.output[0].shape[1])

    with tf.GradientTape() as tape:
        logits, v = model(states, training=True)

        advantage = discounted_rewards - v # stop_gradientは不要 (vの勾配はvalue_lossでのみ使われるため)
        
        if standardize:
            advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)

        # AdvantageはActorの学習にのみ使い、勾配は流さない
        advantage_no_grad = tf.stop_gradient(advantage)

        action_probs = tf.nn.softmax(logits)
        log_probs = tf.math.log(tf.clip_by_value(action_probs, 1e-10, 1.0))
        selected_log_probs = tf.reduce_sum(log_probs * onehot_actions, axis=1, keepdims=True)
        
        policy_loss = -selected_log_probs * advantage_no_grad

        value_loss = tf.keras.losses.huber(discounted_rewards, v)

        entropy = -tf.reduce_sum(action_probs * log_probs, axis=1, keepdims=True)
        
        total_loss = policy_loss + value_loss_weight * value_loss - entropy_weight * entropy
        total_loss = tf.reduce_mean(total_loss)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # ★★★ 報告用のエントロピーを正しい正の値にする ★★★
    return total_loss.numpy(), tf.reduce_mean(entropy).numpy()

def train(env,model):
    # --- ハイパーパラメータ (修正済み) ---
    standardize = True
    total_episodes = 1000
    gamma = 0.99
    # ★ 学習率を下げる (非常に重要)
    lr = 1e-4 #7e-4
    clipnorm = 0.5
    value_loss_weight = 0.5
    # ★ エントロピー係数を上げる
    entropy_weight = 0.02 #0.01

    print('A2C 通常版')
    print('standardize =',standardize)    
    print('total_episodes =',total_episodes)
    print('gamma =',gamma)
    print('lr =',lr)
    print('clipnorm =',clipnorm)
    print('value_loss_weight =',value_loss_weight)
    print('entropy_weight =',entropy_weight)

    optimizer = Adam(learning_rate=lr, clipnorm=clipnorm)

    print("--- 学習開始 ---")
    start_time = time.time()
    all_rewards = []
    all_episode_steps = []
    all_losses = [] 
    all_entropies = []
    total_steps = 0
    
    for episode in range(1,total_episodes+1):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        episode_step_count = 0
        episode_experiences = []

        # 1エピソード実行
        while not (done or truncated):
            episode_step_count += 1
            action = get_action(model, state)
            n_state, reward, done, truncated, _ = env.step(action)

            total_reward += reward

            episode_experiences.append({
                "state": state,
                "action": action,
                "reward": reward,
                "n_state": n_state,
                "done": done,
            })
            state = n_state

        # エピソード終了時に学習
        if len(episode_experiences) > 0:
            loss, entropy = update(model, optimizer, episode_experiences, gamma, value_loss_weight, entropy_weight, standardize)
            all_losses.append(loss)
            all_entropies.append(entropy)

        all_rewards.append(total_reward)
        all_episode_steps.append(episode_step_count)
        total_steps += episode_step_count


        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(all_rewards[-20:])
            avg_episode_steps = np.mean(all_episode_steps[-20:])
            print(f'Episode:{episode+1}/{total_episodes}|Step:{total_steps}|AvgSteps:{avg_episode_steps:.1f}|Avg Reward (last 20):{avg_reward:.1f}|Loss:{loss:.4f}|Entropy:{entropy:.4f}')
            if avg_reward > 475: # CartPole-v1のクリア基準
                print("Environment solved!")
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

    model_file = 'lunarlander-a2c-episode.weights.h5'
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
    imageio.mimsave('lunarlander-a2c-episode.gif', frames, fps=30)
    print(f"GIFを'lunarlander-a2c-episode.gif'に保存しました。最終報酬: {test_reward:.2f}")

