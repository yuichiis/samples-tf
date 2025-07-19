import tensorflow as tf
import numpy as np
import collections
import random
import gymnasium as gym

class QNetwork(tf.keras.Model):
    def __init__(self, num_actions, hidden_units=(64, 64)):
        super(QNetwork, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(units, activation='relu') for units in hidden_units]
        self.output_layer = tf.keras.layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return self.output_layer(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states), np.array(dones, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self,
                 num_states,
                 num_actions,
                 learning_rate=0.001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay_steps=10000,
                 replay_buffer_capacity=100000,
                 batch_size=64,
                 target_update_frequency=1000):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.policy_network = QNetwork(num_actions)
        self.target_network = QNetwork(num_actions)

        # ターゲットネットワークを初期化
        dummy_state = tf.random.normal([1, num_states])
        self.policy_network(dummy_state)
        self.target_network(dummy_state)
        self.update_target_network()

        self.train_step_counter = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            q_values = self.policy_network(np.expand_dims(state, axis=0))
            return np.argmax(q_values[0])

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update_target_network(self):
        self.target_network.set_weights(self.policy_network.get_weights())

    def update_epsilon(self):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-self.train_step_counter / self.epsilon_decay_steps)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        with tf.GradientTape() as tape:
            q_values = self.policy_network(states)
            q_value = tf.gather_nd(q_values, tf.stack([tf.range(self.batch_size), actions], axis=1))

            next_q_values_target = self.target_network(next_states)
            next_q_value_target = tf.reduce_max(next_q_values_target, axis=1)

            target_q_value = rewards + self.gamma * next_q_value_target * (1 - dones)
            loss = tf.keras.losses.MSE(target_q_value, q_value)

        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))

        self.train_step_counter += 1
        self.update_epsilon()

        if self.train_step_counter % self.target_update_frequency == 0:
            self.update_target_network()

        return loss.numpy()

# 使用例
if __name__ == '__main__':
    # 環境の準備 (例として OpenAI Gym の CartPole-v1 を使用)
    env = gym.make('CartPole-v1')
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # DQNエージェントの初期化
    agent = DQNAgent(num_states=num_states, num_actions=num_actions)

    # ハイパーパラメータ
    episodes = 500
    max_steps_per_episode = 200
    reward_history = []
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            episode_reward += reward
            state = next_state
            if done:
                break
        reward_history.append(episode_reward)
        print(f"Episode: {episode + 1}, Reward: {episode_reward}, Epsilon: {agent.epsilon}, Replay Buffer Size: {len(agent.replay_buffer)}")

    print("Training finished.")
    env.close()

    # 学習済みモデルの保存 (必要に応じて)
    # agent.policy_network.save_weights('dqn-cartpole.weights.h5')