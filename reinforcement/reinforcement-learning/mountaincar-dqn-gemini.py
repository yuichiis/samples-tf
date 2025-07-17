import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque
import os
import imageio

# --- ハイパーパラメータ設定 ---
ENV_NAME = "MountainCar-v0"

GAMMA = 0.99                # 割引率
LEARNING_RATE = 0.001       # 学習率

MEMORY_SIZE = 10000         # 経験再生バッファのサイズ
BATCH_SIZE = 64             # バッチサイズ

EPSILON_START = 1.0         # ε-greedy法のεの初期値
EPSILON_END = 0.01          # εの最終値
EPSILON_DECAY = 0.995       # εの減衰率

TARGET_UPDATE_FREQ = 10     # ターゲットネットワークを更新する頻度（エピソードごと）

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # 経験再生のためのメモリ（dequeは高速な両端キュー）
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        # ε-greedy法のためのε
        self.epsilon = EPSILON_START
        
        # メインのQネットワーク
        self.q_network = self._build_model()
        # ターゲットネットワーク（学習を安定させるため）
        self.target_q_network = self._build_model()
        # 最初は同じ重みで初期化
        self.update_target_network()

    def _build_model(self):
        """ニューラルネットワークモデルを構築"""
        model = Sequential([
            Dense(24, activation='relu', input_shape=(self.state_size,)),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear') # Q値なので活性化関数はlinear
        ])
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
        return model

    def update_target_network(self):
        """メインネットワークの重みをターゲットネットワークにコピー"""
        self.target_q_network.set_weights(self.q_network.get_weights())
        # print("--- Target Network Updated ---")

    def remember(self, state, action, reward, next_state, done):
        """経験をメモリに保存"""
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """ε-greedy法に基づいて行動を選択"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # ランダムに行動
        else:
            # Qネットワークが予測したQ値が最大となる行動を選択
            q_values = self.q_network.predict(state, verbose=0)
            return np.argmax(q_values[0])

    def replay(self):
        """経験再生によりネットワークを学習"""
        if len(self.memory) < BATCH_SIZE:
            return # メモリがバッチサイズより小さい場合は学習しない
        
        if(len(self.memory)>MEMORY_SIZE):
            print('memory=',len(self.memory))

        # メモリからランダムにバッチをサンプリング
        minibatch = random.sample(self.memory, BATCH_SIZE)
        
        states = np.array([experience[0].flatten() for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3].flatten() for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # ターゲットQ値を計算
        # 1. まずターゲットネットワークで次の状態のQ値を予測
        target_q_next = self.target_q_network.predict(next_states, verbose=0)
        
        # 2. ベルマン方程式に基づいてターゲットを計算
        #    target = reward (もしエピソード終了)
        #    target = reward + gamma * max(Q(next_state)) (それ以外)
        targets = rewards + GAMMA * np.amax(target_q_next, axis=1) * (1 - dones)
        
        # 現在のQ値を取得し、選択されたアクションの部分だけターゲットで更新
        current_q_values = self.q_network.predict(states, verbose=0)
        for i, action in enumerate(actions):
            current_q_values[i][action] = targets[i]
            
        # ネットワークを学習
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0)

    def decay_epsilon(self):
        """εを減衰させる"""
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY


def train(env,agent):
    episodes = 500
    scores = []

    for e in range(episodes):
        state, info = env.reset()
        max_pos = state[0]
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        done = False
        step = 0
        
        # Gymnasiumはterminatedとtruncatedを返す
        terminated, truncated = False, False

        while not (terminated or truncated):
            step += 1
            # 行動を選択
            action = agent.choose_action(state)
            
            # 環境中で行動を実行
            next_state, reward, terminated, truncated, info = env.step(action)
            # MountainCarはゴール地点の速度が重要なので、報酬を調整すると学習が早まることがある
            # 例: positionが上がったら少しだけプラスの報酬を与える
            if next_state[0] > max_pos:
                max_pos = next_state[0]
            if terminated or truncated:
                reward = reward + 200 * max_pos
            if max_pos > 0.5:
                reward = reward + 200
            #reward = reward + 300 * (abs(next_state[1]))
            #print('pos:',next_state[0],'max_pos',max_pos)

            next_state = np.reshape(next_state, [1, state_size])
            total_reward += reward # ここでのrewardは元の-1
            
            # 経験をメモリに保存
            agent.remember(state, action, reward, next_state, terminated)
            
            state = next_state

            if terminated or truncated:
                scores.append(total_reward)
                pos,vel = state[0]
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f},step:{step},maxpos:{max_pos:.2f},term:{terminated},trunc:{truncated}",end="")
                # 最近10エピソードの平均スコアを表示
                if len(scores) > 10:
                    print(f" Avg Score: {np.mean(scores[-10:]):.2f}")
                else:
                    print()
                break

        # 経験再生で学習
        agent.replay()
        
        # εを減衰
        agent.decay_epsilon()
        
        # 一定エピソードごとにターゲットネットワークを更新
        if (e + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
    

if __name__ == "__main__":
    # 環境とエージェントの初期化
    # render_mode="human" をつけると描画される
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    model_file = 'mountaincar-dqn-model.weights.h5'
    if os.path.isfile(model_file):
        agent.q_network.load_weights(model_file)
        agent.update_target_network()
    else:
        train(env,agent)
        agent.q_network.save(model_file)

    env.close()

    # 学習後のエージェントの性能をテスト（描画あり）
    print("\n--- Testing Trained Agent ---")
    env = gym.make(ENV_NAME, render_mode='rgb_array')
    for i in range(1):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        frames = []
        done = False
        total_reward = 0
        while not done:
            frames.append(env.render())
            action = np.argmax(agent.q_network.predict(state, verbose=0)[0]) # ε=0で行動
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, [1, state_size])
            total_reward += reward
        print(f"Test Episode {i+1}, Score: {total_reward}")
    env.close()
    imageio.mimsave('mountaincar_dqn_trained.gif', frames, fps=30)
    print(f"GIFを'mountaincar_dqn_trained.gif'に保存しました。最終報酬: {total_reward:.2f}")
