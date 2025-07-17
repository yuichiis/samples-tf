import os
import collections
import random
#import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import gymnasium as gym
import imageio

# GPUメモリの動的確保（必要な場合）
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- ハイパーパラメータ ---
ENV_NAME = "LunarLander-v3"
GAMMA = 0.99  # 割引率
LEARNING_RATE = 1e-3  # 学習率

BUFFER_SIZE = 100_000  # リプレイバッファの最大サイズ
BATCH_SIZE = 64      # バッチサイズ
TARGET_UPDATE_FREQ = 500  # ターゲットネットワークを更新する頻度（ステップ数）

EPSILON_START = 1.0       # ε-greedyの初期値
EPSILON_END = 0.02        # ε-greedyの最終値
EPSILON_DECAY_STEPS = 100_000 # εが最終値に達するまでのステップ数

MAX_EPISODES = 1000 # 最大エピソード数
MAX_STEPS = 300000  # 最大総ステップ数

# --- リプレイバッファ ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        """経験をバッファに保存"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """バッファからランダムにバッチをサンプリング"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# --- Qネットワークモデルの定義 ---
def create_q_model(state_shape, num_actions):
    """Q値を予測するニューラルネットワークを作成"""
    model = models.Sequential([
        layers.Input(shape=state_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_actions, activation='linear') # Q値は任意の値なので活性化関数はlinear
    ])
    return model
    

def train(env,state_shape,num_actions,q_network):
    # ターゲットネットワークを作成
    target_q_network = create_q_model(state_shape, num_actions)
    # ターゲットネットワークの重みをメインネットワークと同じにする
    target_q_network.set_weights(q_network.get_weights())

    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    # Huber損失は外れ値に強いMSEのようなもの
    loss_function = losses.Huber()

    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    epsilon = EPSILON_START
    total_steps = 0
    episode_rewards = []
    episode_steps = []

    print("学習を開始します...")

    # 2. 学習ループ
    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        episode_step = 0
        done = False

        while not done:
            total_steps += 1
            episode_step += 1

            # 3. ε-greedy法による行動選択
            if np.random.rand() < epsilon:
                # 探索: ランダムに行動を選択
                action = env.action_space.sample()
            else:
                # 活用: Qネットワークが予測する最善の行動を選択
                q_values = q_network(np.expand_dims(state, axis=0))
                action = tf.argmax(q_values[0]).numpy()

            # 4. 環境中で行動を実行
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # 5. リプレイバッファに経験を保存
            replay_buffer.store(state, action, reward, next_state, done)

            state = next_state

            # 6. リプレイバッファからサンプリングしてネットワークを更新
            if len(replay_buffer) > BATCH_SIZE:
                # バッファからミニバッチをサンプリング
                experiences = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = map(np.array, zip(*experiences))

                # ターゲットQ値の計算
                # donesがTrueの場合、将来の価値は0になる
                future_rewards = target_q_network(next_states)
                target_q_values = rewards + (1 - dones) * GAMMA * tf.reduce_max(future_rewards, axis=1)

                # 勾配を計算してメインネットワークを更新
                with tf.GradientTape() as tape:
                    # 実際に取った行動に対応するQ値のみを取得
                    q_values = q_network(states)
                    action_masks = tf.one_hot(actions, num_actions)
                    selected_action_q_values = tf.reduce_sum(q_values * action_masks, axis=1)

                    # 損失を計算
                    loss = loss_function(target_q_values, selected_action_q_values)

                # 勾配を適用
                grads = tape.gradient(loss, q_network.trainable_variables)
                optimizer.apply_gradients(zip(grads, q_network.trainable_variables))

            # 7. ターゲットネットワークの更新
            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_q_network.set_weights(q_network.get_weights())

            # 8. εの値を減衰させる
            epsilon = max(
                EPSILON_END,
                EPSILON_START - (EPSILON_START - EPSILON_END) * (total_steps / EPSILON_DECAY_STEPS)
            )
            
            if total_steps > MAX_STEPS:
                break
        
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
        mean_reward = np.mean(episode_rewards[-100:])
        mean_step = np.mean(episode_steps[-100:])

        print(f"エピソード: {episode + 1}, ステップ: {episode_step}, 総ステップ: {total_steps}, 報酬: {episode_reward:.2f}, "
              f"過去100回の平均報酬: {mean_reward:.2f}, ε: {epsilon:.4f}")
        
        # 学習終了条件 (過去100エピソードの平均報酬が200を超えたら成功とみなす)
        if mean_reward >= 200:
            print(f"\n環境クリア！ {episode + 1}エピソードで解決しました。")
            break
            
        if total_steps > MAX_STEPS:
            print("\n最大ステップ数に達しました。学習を終了します。")
            break



# --- メイン処理 ---
if __name__ == "__main__":
    # 1. 環境とモデルの初期化
    env = gym.make(ENV_NAME)
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n

    # メインネットワークを作成
    q_network = create_q_model(state_shape, num_actions)

    model_file = 'lunarlander-dqn.weights.h5'
    if os.path.isfile(model_file):
        q_network.load_weights(model_file)
    else:
        train(env,state_shape,num_actions,q_network)

        # 9. 学習済みモデルの保存と評価
        q_network.save(model_file)
    env.close()

    print("\n学習済みモデルでエージェントを動かし、GIFを生成します...")

    # GIF生成
    trained_env = gym.make(ENV_NAME, render_mode='rgb_array')
    state, _ = trained_env.reset()
    frames = []
    done = False
    episode_reward = 0
    while not done:
        frames.append(trained_env.render())
        q_values = q_network(np.expand_dims(state, axis=0))
        action = tf.argmax(q_values[0]).numpy()
        state, reward, terminated, truncated, _ = trained_env.step(action)
        done = terminated or truncated
        episode_reward += reward

    trained_env.close()
    imageio.mimsave('lunarlander-dqn.gif', frames, fps=30)
    print(f"GIFを'lunarlander-dqn.gif'に保存しました。最終報酬: {episode_reward:.2f}")
