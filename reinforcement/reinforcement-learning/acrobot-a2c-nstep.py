import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import gymnasium as gym
import matplotlib.pyplot as plt
import time

# === モデルの定義 ===
def create_a2c_model(input_shape, num_actions):
    """Actor-Criticモデルを生成する"""
    input_layer = keras.layers.Input(shape=input_shape)
    common_layer = keras.layers.Dense(64, activation="relu")(input_layer)
    common_layer = keras.layers.Dense(64, activation="relu")(common_layer)
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

# === 学習関数 (修正済み) ===
def train(model, optimizer, experiences, gamma, value_loss_weight, entropy_weight, standardize):
    states = np.asarray([e["state"] for e in experiences], dtype=np.float32)
    actions = np.asarray([e["action"] for e in experiences], dtype=np.int32)
    rewards = np.asarray([e["reward"] for e in experiences], dtype=np.float32)
    dones = np.asarray([e["done"] for e in experiences], dtype=np.bool_)

    if not dones[-1]:
        last_state = experiences[-1]["n_state"].reshape((1, -1))
        _, last_v = model(last_state)
        last_v = last_v.numpy()[0, 0]
    else:
        last_v = 0.0

    discounted_rewards = []
    G = last_v
    for r, done in zip(rewards[::-1], dones[::-1]):
        if done:
            G = 0
        G = r + gamma * G
        discounted_rewards.append(G)
    discounted_rewards.reverse()
    discounted_rewards = np.array(discounted_rewards, dtype=np.float32).reshape(-1, 1)

    # # ベースライン処理
    # discounted_rewards -= np.mean(discounted_rewards)  # 報酬の平均を引く

    onehot_actions = tf.one_hot(actions, model.output[0].shape[1])

    with tf.GradientTape() as tape:
        logits, v = model(states, training=True)

        # ★★★ 修正点 1: Advantage計算時にCriticの勾配を止める ★★★
        # これでAdvantageの計算がCriticの学習に影響を与えないことが保証される
        advantage = discounted_rewards - tf.stop_gradient(v)
        
        if(standardize):
            advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)

        action_probs = tf.nn.softmax(logits)
        selected_action_probs = tf.reduce_sum(action_probs * onehot_actions, axis=1, keepdims=True)
        selected_action_probs = tf.clip_by_value(selected_action_probs, 1e-10, 1.0)
        
        # Policy Lossは正規化されたAdvantageを使う
        # tf.stop_gradientは不要になったが、念のため残しても害はない
        policy_loss = -tf.math.log(selected_action_probs) * advantage

        # Critic (Value) Lossは、Actorの学習とは独立して、vがG_tに近づくように学習する
        value_loss = tf.keras.losses.huber(discounted_rewards, v)

        action_probs_clipped = tf.clip_by_value(action_probs, 1e-10, 1.0)
        entropy = -tf.reduce_sum(action_probs_clipped * tf.math.log(action_probs_clipped), axis=1, keepdims=True)

        total_loss = policy_loss + value_loss_weight * value_loss + entropy_weight * entropy
        total_loss = tf.reduce_mean(total_loss)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss.numpy(), tf.reduce_mean(entropy).numpy()

# === メイン処理 ===
if __name__ == '__main__':
    env = gym.make("Acrobot-v1")
    obs_shape = env.observation_space.shape
    nb_actions = env.action_space.n
    model = create_a2c_model(obs_shape, nb_actions)
    model.summary()
    
    # --- ハイパーパラメータ (修正済み) ---
    standardize = True
    total_timesteps = 180000
    n_steps = 512#512#32
    gamma = 0.99
    lr = 1e-3
    clipnorm = 0.5
    value_loss_weight = 0.5#0.25
    entropy_weight = 0.01#0.02

    print('A2C 通常版')
    print('standardize =',standardize)    
    print('total_timesteps =',total_timesteps)
    print('n_steps =',n_steps)
    print('gamma =',gamma)
    print('lr =',lr)
    print('clipnorm =',clipnorm)
    print('value_loss_weight =',value_loss_weight)
    print('entropy_weight =',entropy_weight)



    optimizer = Adam(learning_rate=lr, clipnorm=clipnorm)

    print("--- 学習開始 ---")
    start_time = time.time()
    all_rewards = []
    all_losses = [] 
    all_entropies = []
    
    experiences = []
    episode_count = 0
    update_count = 0
    episode_reward_sum = 0
    
    state, _ = env.reset()

    for global_step in range(1, total_timesteps + 1):
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
            episode_count += 1
            episode_reward_sum = 0
            state, _ = env.reset()

        if len(experiences) >= n_steps:
            loss, entropy = train(model, optimizer, experiences, gamma, value_loss_weight, entropy_weight, standardize)
            all_losses.append(loss)
            all_entropies.append(entropy)
            experiences = []
            update_count += 1

        if (global_step % (n_steps*10) == 0) or global_step == total_timesteps:
            avg_reward = np.mean(all_rewards[-20:])
            last_loss = all_losses[-1] if all_losses else 0
            last_entropy = all_entropies[-1] if all_entropies else 0
            print(f'Update#{update_count} | Step {global_step}/{total_timesteps//1000}k | Ep {episode_count} | Avg Reward (last 20): {avg_reward:.1f} | Loss: {last_loss:.3f} | Entropy: {last_entropy:.3f}')
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
    
    print("\n--- テスト実行 ---")
    env_render = gym.make("Acrobot-v1", render_mode="human")
    for i in range(5):
        state, _ = env_render.reset()
        done, truncated = False, False
        test_reward = 0
        while not (done or truncated):
            action = get_best_action(model, state)
            state, reward, done, truncated, _ = env_render.step(action)
            test_reward += reward
        print(f"Test Episode {i+1}, Total Reward: {test_reward}")
    env_render.close()
