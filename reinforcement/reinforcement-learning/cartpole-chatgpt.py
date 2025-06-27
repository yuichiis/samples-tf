import tensorflow as tf
import numpy as np
import random
from collections import deque

# Qネットワークの定義（ポリシーとターゲットで共通の構造）
class QNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        # 隠れ層（例：64ユニット×2層）
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        # 出力層：各行動のQ値を出力
        self.out = tf.keras.layers.Dense(num_actions, activation=None)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.out(x)

# DQNエージェントの定義
class DQNAgent:
    def __init__(self, state_size, num_actions, gamma=0.99, learning_rate=0.001,
                 buffer_size=10000, batch_size=64, target_update_freq=1000):
        self.state_size = state_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0

        # リプレイバッファの準備
        self.replay_buffer = deque(maxlen=buffer_size)

        # ポリシーネットワークとターゲットネットワークの生成
        self.policy_net = QNetwork(num_actions)
        self.target_net = QNetwork(num_actions)
        # 初期はターゲットネットワークの重みをポリシーネットワークと同じに設定
        self.policy_net(np.random.rand(1, state_size))
        self.target_net(np.random.rand(1, state_size))
        print(self.target_net.get_weights())
        self.target_net.set_weights(self.policy_net.get_weights())

        # オプティマイザの設定
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    # 経験をリプレイバッファに格納
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    # ε-greedy法に基づく行動選択
    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return random.randrange(self.num_actions)
        else:
            # 状態をバッチ次元追加して予測
            q_values = self.policy_net(np.expand_dims(state, axis=0))
            return np.argmax(q_values.numpy()[0])

    # ミニバッチを用いた学習ステップ
    def train_step(self):
        # バッチサイズ未満の場合は学習しない
        if len(self.replay_buffer) < self.batch_size:
            return

        # リプレイバッファからランダムにサンプル
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        # 次状態でのQ値をターゲットネットワークで計算し、最大値を取得
        target_q = self.target_net(next_states)
        max_target_q = np.max(target_q.numpy(), axis=1)
        # TDターゲットの計算（doneがTrueの場合は次状態の価値は0）
        target = rewards + (1 - dones) * self.gamma * max_target_q

        with tf.GradientTape() as tape:
            q_values = self.policy_net(states)
            # 選択した行動のQ値のみを抽出（one-hot表現を利用）
            actions_onehot = tf.one_hot(actions, self.num_actions)
            q_action = tf.reduce_sum(q_values * actions_onehot, axis=1)
            loss = tf.keras.losses.MSE(target, q_action)

        # 勾配の計算とパラメータの更新
        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

        self.step_count += 1
        # ターゲットネットワークの重み更新
        if self.step_count % self.target_update_freq == 0:
            self.target_net.set_weights(self.policy_net.get_weights())

# 学習ループの例（ダミーの環境データを使用）
if __name__ == "__main__":
    # 状態の次元数と行動数の例（例：状態サイズ=4, 行動数=2）
    state_size = 4
    num_actions = 2

    agent = DQNAgent(state_size, num_actions)

    num_episodes = 500
    for episode in range(num_episodes):
        # ダミーの初期状態
        state = np.random.rand(state_size)
        done = False
        total_reward = 0

        while not done:
            # ε=0.1 のε-greedy法による行動選択
            action = agent.choose_action(state, epsilon=0.1)
            # 環境から得られるダミーの次状態、報酬、エピソード終了判定
            next_state = np.random.rand(state_size)
            reward = np.random.rand()
            done = np.random.rand() < 0.1  # 10%の確率で終了

            # 経験の保存と学習
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

        print(f"Episode {episode+1}: Total Reward = {total_reward}")
