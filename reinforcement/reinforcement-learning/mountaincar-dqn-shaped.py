import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random

# 環境設定
env = gym.make("MountainCar-v0")
obs_shape = env.observation_space.shape
n_actions = env.action_space.n

# Qネットワークの定義
def build_model():
    model = tf.keras.Sequential([
        layers.Input(shape=obs_shape),
        layers.Dense(24, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(n_actions, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')
    return model

# パラメータ
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 400
batch_size = 64
memory = deque(maxlen=20000)

# モデルの初期化
model = build_model()
target_model = build_model()
target_model.set_weights(model.get_weights())

## 報酬 shaping
#def shaped_reward(state, reward):
#    position, velocity = state
#    reward += (position + 0.5) * 2  # positionが -0.5〜0.5 → reward 0〜2 へ補正
#    return reward

def potential_energy(position):
    return np.sin(3 * position)

def kinetic_energy(velocity):
    return 0.5 * velocity ** 2

def shaped_reward(state, reward, done):
    position, velocity = state

    # 現在のエネルギー
    E_p = potential_energy(position)
    E_k = kinetic_energy(velocity)
    total_energy = E_p + E_k

    # 初期化（エピソード単位で）
    if not hasattr(shaped_reward, "last_energy"):
        shaped_reward.last_energy = total_energy

    # エネルギーの増加量を報酬とする
    energy_gain = total_energy - shaped_reward.last_energy
    shaped_reward.last_energy = total_energy

    # ゴール達成ボーナス
    if done and position >= 0.5:
        return reward + energy_gain * 10 + 100

    return reward + energy_gain * 10  # スケーリング係数調整可能


# 経験の記録
def remember(s, a, r, s2, done):
    memory.append((s, a, r, s2, done))

# 経験から学習
def replay():
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    states, targets = [], []

    for state, action, reward, next_state, done in minibatch:
        target = model.predict(np.array([state]), verbose=0)[0]
        if done:
            target[action] = reward
        else:
            next_q = np.amax(target_model.predict(np.array([next_state]), verbose=0)[0])
            target[action] = reward + gamma * next_q
        states.append(state)
        targets.append(target)
    
    model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

# メインループ
for e in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for t in range(200):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model.predict(np.array([state]), verbose=0)
            action = np.argmax(q_values[0])

        next_state, reward, done, truncated, _ = env.step(action)

        reward = shaped_reward(next_state, reward, done)
        remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if (done or truncated):
            print(f"Episode {e+1}/{episodes} - Score: {total_reward:.2f}, Epsilon: {epsilon:.2f}")
            break

        replay()
    
    # ターゲットネットワークの更新
    if e % 10 == 0:
        target_model.set_weights(model.get_weights())

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

env.close()
