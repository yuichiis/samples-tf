import numpy as np
from collections import deque
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import gymnasium as gym
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import imageio


# === 方策関数 (変更なし) ===
def get_action(model, state, nb_actions):
    # (関数名をより分かりやすく変更)
    logits, _ = model(state.reshape((1, -1)))
    probs = tf.nn.softmax(logits)
    return np.random.choice(nb_actions, p=probs[0].numpy())

# === 学習関数 (大幅に修正) ===
def update(model, optimizer, experiences, gamma, value_loss_weight, entropy_weight):
    # 経験からバッチデータを作成
    states = np.asarray([e["state"] for e in experiences])
    actions = np.asarray([e["action"] for e in experiences])
    rewards = np.asarray([e["reward"] for e in experiences])
    next_states = np.asarray([e["n_state"] for e in experiences])
    dones = np.asarray([e["done"] for e in experiences])

    # === 割引報酬和 (G_t) の計算 ===
    # Criticを使って最後の状態の価値をブートストラップする
    _, last_v = model(next_states[-1].reshape((1, -1)))
    last_v = last_v.numpy()[0, 0]
    
    # G_tを後ろから計算していく
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

    # one-hotアクションベクトル
    onehot_actions = tf.one_hot(actions, nb_actions)

    # === 勾配を計算 ===
    with tf.GradientTape() as tape:
        logits, v = model(states, training=True)

        # --- アドバンテージ A(s,a) = G_t - V(s) ---
        # 勾配計算にvの影響を与えないようにする
        advantage = discounted_rewards - tf.stop_gradient(v)
        
        # ★★★ アドバンテージの正規化を追加 ★★★
        # advantageから平均を引き、標準偏差で割る
        advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)
        # 1e-8は、標準偏差が0の場合にゼロ除算になるのを防ぐための小さな値(epsilon)

        # --- Actor (Policy) Loss ---
        # log(π(a|s)) * A(s,a)
        action_probs = tf.nn.softmax(logits)
        selected_action_probs = tf.reduce_sum(action_probs * onehot_actions, axis=1, keepdims=True)
        selected_action_probs = tf.clip_by_value(selected_action_probs, 1e-10, 1.0)
        policy_loss = -tf.math.log(selected_action_probs) * advantage

        # --- Critic (Value) Loss ---
        # V(s) が G_t に近づくように学習 (Huber損失が安定)
        value_loss = tf.keras.losses.huber(discounted_rewards, v)

        # --- Entropy Loss ---
        # H(π) = -Σ p(a|s) * log(p(a|s))
        # 0でlogを取るのを防ぐ
        action_probs = tf.clip_by_value(action_probs, 1e-10, 1.0)
        entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs), axis=1, keepdims=True)

        # --- Total Loss ---
        # policy_lossとvalue_lossは最小化、entropyは最大化（なのでマイナスをかける）
        total_loss = policy_loss + value_loss_weight * value_loss + entropy_weight * entropy
        total_loss = tf.reduce_mean(total_loss)

    # 勾配を計算し、モデルを更新
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss.numpy(), tf.reduce_mean(entropy).numpy()

def train(env,model):
    # === ハイパーパラメータ ===
    iterations = 1500#400
    gamma = 0.99  # 割引率
    lr = 1e-3     # 学習率 (下げた)
    value_loss_weight = 0.5#0.25  # 0.5から0.25へ下げる
    entropy_weight = 0.01#0.02 # 0.01から少し上げる

    optimizer = Adam(learning_rate=lr, clipnorm=0.5)  # clipnormを追加 (0.5や1.0が一般的)

    # === 学習ループ (修正) ===
    print("--- 学習開始 ---")
    start_time = time.time()
    all_rewards = []
    all_losses = []
    all_entropies = []

    for episode in range(iterations):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        episode_experiences = []

        # 1エピソード実行
        while not (done or truncated):
            action = get_action(model, state, nb_actions)
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
            loss, entropy = update(model, optimizer, episode_experiences, gamma, value_loss_weight, entropy_weight)
            all_losses.append(loss)
            all_entropies.append(entropy)

        all_rewards.append(total_reward)

        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(all_rewards[-20:])
            print(f'Episode {episode+1}/{iterations} | Avg Reward (last 20): {avg_reward:.1f} | Loss: {loss:.4f} | Entropy: {entropy:.4f}')
            if avg_reward > 475: # CartPole-v1のクリア基準
                print("Environment solved!")
                break

    print("--- 学習終了 ---")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"実行時間: {execution_time:.4f}秒")

    # === 結果のプロット ===
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(all_rewards)
    plt.title('Total Reward per Episode')
    plt.ylabel('Reward')
    # 性能の傾向を見やすくするために移動平均もプロット
    moving_avg = np.convolve(all_rewards, np.ones(20)/20, mode='valid')
    plt.plot(moving_avg)
    plt.legend(['Reward', 'Moving Average (20 ep)'])


    plt.subplot(2, 1, 2)
    plt.plot(all_losses, label='Loss', alpha=0.7)
    plt.plot(all_entropies, label='Entropy', alpha=0.7)
    plt.title('Loss and Entropy')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_best_action(model, state, nb_actions):
    logits, _ = model(state.reshape((1, -1)))
    # 確率が最大のアクションのインデックスを返す
    return tf.argmax(logits[0]).numpy()

# === メイン処理 ===
if __name__ == '__main__':
    # === 環境の準備 ===
    env = gym.make("CartPole-v1")
    obs_shape = env.observation_space.shape
    nb_actions = env.action_space.n

    # === モデルの定義 (変更なし) ===
    c = input_ = keras.layers.Input(shape=obs_shape)
    c = keras.layers.Dense(64, activation="relu")(c) # 少し層を厚くすると性能が上がることがある
    c = keras.layers.Dense(64, activation="relu")(c)
    actor_layer = keras.layers.Dense(nb_actions, activation="linear")(c)
    critic_layer = keras.layers.Dense(1, activation="linear")(c)

    model = keras.Model(input_, [actor_layer, critic_layer])
    model.summary()
    model_file = 'cartpole-a2c-episode-model.weights.h5'
    if os.path.isfile(model_file):
        model.load_weights(model_file)
    else:
        # === 学習実行 ===
        train(env,model)
        model.save(model_file)

    # === テスト実行 ===
    env_render = gym.make("CartPole-v1", render_mode="rgb_array")
    for episode in range(1):
        state, _ = env_render.reset()
        frames = []
        done = False
        truncated = False
        total_reward = 0
        while not (done or truncated):
            frames.append(env_render.render())
            action = get_best_action(model, state, nb_actions)
            state, reward, done, truncated, _ = env_render.step(action)
            total_reward += reward
        print(f"Test Episode {episode+1}, Total Reward: {total_reward}")
    env_render.close()
    imageio.mimsave('cartpole-a2c-episode-trained.gif', frames, fps=30)
    print(f"GIFを'cartpole-a2c-episode-trained.gif'に保存しました。最終報酬: {total_reward:.2f}")
