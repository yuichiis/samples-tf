import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 環境の作成
env = gym.make('CartPole-v1')

# パラメータの設定
learning_rate = 0.01
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 1000

# Qネットワークの構築
model = keras.Sequential([
    keras.layers.Dense(24, activation='relu', input_shape=(4,)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(2, activation='linear')  # 行動数（左右）
])

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# Q学習の実装
for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, 4])
    total_reward = 0

    while True:
        # ε-greedy法による行動選択
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            q_values = model(state)
            action = np.argmax(q_values.numpy()[0])

        # 行動の実行と次の状態、報酬の取得
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        total_reward += reward

        # Q値の更新
        if done:
            target = reward
        else:
            target = reward + gamma * np.max(model(next_state).numpy()[0])

        with tf.GradientTape() as tape:
            q_values = model(state)
            q_value = q_values[0][action]
            loss = tf.square(target - q_value)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        state = next_state

        if done:
            break

    # εの減衰
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if (episode + 1) % 100 == 0:
        print(f'Episode {episode + 1}, Total Reward: {total_reward}')

# 学習済みモデルの評価
total_rewards = []
for _ in range(100):
    state = env.reset()
    state = np.reshape(state, [1, 4])
    total_reward = 0
    while True:
        q_values = model(state)
        action = np.argmax(q_values.numpy()[0])
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        total_reward += reward
        state = next_state
        if done:
            break
    total_rewards.append(total_reward)

print(f'Average reward over 100 episodes: {np.mean(total_rewards)}')

env.close()
