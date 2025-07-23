import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

# --- 環境設定 ---
env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

# --- ハイパーパラメータ ---
gamma = 0.99
actor_lr = 0.0005
critic_lr = 0.001
entropy_coef = 0.0001

# --- Actorモデル ---
class Actor(keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = layers.Dense(128, activation="relu")
        self.fc2 = layers.Dense(128, activation="relu")
        self.mean = layers.Dense(action_dim)
        self.log_std = tf.Variable(initial_value=-0.5 * tf.ones(action_dim), trainable=True)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        mean = self.mean(x)
        std = tf.exp(self.log_std)
        return mean, std

# --- Criticモデル ---
class Critic(keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = layers.Dense(128, activation="relu")
        self.fc2 = layers.Dense(128, activation="relu")
        self.v = layers.Dense(1)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.v(x)

# --- モデルとオプティマイザ初期化 ---
actor = Actor()
critic = Critic()
actor_optimizer = keras.optimizers.Adam(learning_rate=actor_lr)
critic_optimizer = keras.optimizers.Adam(learning_rate=critic_lr)

# --- アクション選択関数 ---
def get_action(state):
    state = tf.convert_to_tensor([state], dtype=tf.float32)
    mean, std = actor(state)
    dist = tfp.distributions.Normal(mean, std)
    action = dist.sample()
    action = tf.clip_by_value(action, -action_bound, action_bound)
    log_prob = dist.log_prob(action)
    return action[0].numpy(), tf.reduce_sum(log_prob, axis=1)

# --- 学習関数 ---
def train_step(state, action, reward, next_state, done, log_prob):
    state = tf.convert_to_tensor([state], dtype=tf.float32)
    next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
    action = tf.convert_to_tensor([action], dtype=tf.float32)
    reward = tf.convert_to_tensor([reward], dtype=tf.float32)
    done = tf.convert_to_tensor([done], dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        # 値関数
        value = critic(state)
        next_value = critic(next_state)
        target = reward + (1.0 - done) * gamma * next_value
        advantage = target - value

        # actor loss
        mean, std = actor(state)
        dist = tfp.distributions.Normal(mean, std)
        log_prob_new = dist.log_prob(action)
        log_prob_new = tf.reduce_sum(log_prob_new, axis=1)
        entropy = tf.reduce_sum(dist.entropy(), axis=1)
        actor_loss = -log_prob_new * tf.stop_gradient(advantage) - entropy_coef * entropy

        # critic loss
        critic_loss = tf.keras.losses.MSE(target, value)

    # 勾配更新
    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
    del tape

# --- メインループ ---
max_episodes = 500
reward_history = []

for episode in range(max_episodes):
    state = env.reset()[0]
    episode_reward = 0
    done = False

    while not done:
        action, log_prob = get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        train_step(state, action, reward, next_state, done, log_prob)
        state = next_state
        episode_reward += reward

    reward_history.append(episode_reward)
    if episode % 10 == 0:
        avg_reward = np.mean(reward_history[-10:])
        print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

env.close()
