import gymnasium as gym
#import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Hyperparameters
ENV_NAME = "Pendulum-v1"  # Gym Pendulum environment
NUM_ACTIONS = 1
STATE_DIM = 3

GAMMA = 0.99            # Discount factor
LR_ACTOR = 1e-4         # Learning rate for actor
LR_CRITIC = 1e-3        # Learning rate for critic
ENTROPY_COEF = 1e-3     # Entropy regularization coefficient
MAX_EPISODES = 1000
MAX_STEPS = 200         # Max steps per episode

# Actor-Critic Model Definition
class ActorCritic(tf.keras.Model):
    def __init__(self, state_dim, action_dim, action_bound):
        super(ActorCritic, self).__init__()
        self.common = layers.Dense(128, activation="relu")

        # Actor
        self.mu = layers.Dense(action_dim, activation="tanh")
        self.sigma = layers.Dense(action_dim, activation="softplus")

        # Critic
        self.value = layers.Dense(1)

        self.action_bound = action_bound

    def call(self, inputs):
        x = self.common(inputs)
        mu = self.mu(x) * self.action_bound
        sigma = self.sigma(x) + 1e-5
        value = self.value(x)
        return mu, sigma, value

# A2C Agent
class A2CAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        # Build model and optimizers
        self.model = ActorCritic(self.state_dim, self.action_dim, self.action_bound)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_ACTOR)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_CRITIC)

    #def get_action(self, state):
    #    state = state.reshape([1, self.state_dim])
    #    mu, sigma, _ = self.model(state)
    #    dist = tfp.distributions.Normal(mu, sigma)
    #    action = tf.squeeze(dist.sample(1), axis=0)
    #    action = tf.clip_by_value(action, -self.action_bound, self.action_bound)
    #    return action.numpy()[0]
    
    def get_action(self, state):
        state = state.reshape([1, self.state_dim])
        mu, sigma, _ = self.model(state)
        
        # --- tensorflow-probabilityの代替実装 ---
        # 1. 標準正規分布(平均0, 標準偏差1)からノイズを生成
        epsilon = tf.random.normal(shape=tf.shape(mu))
        
        # 2. 再パラメータ化トリックを使ってアクションをサンプリング
        # action = mean + std_dev * noise
        sampled_action = mu + sigma * epsilon
        # ----------------------------------------

        # tf.squeezeは不要になりますが、念のためつけておくと安全です
        # (mu, sigmaのshapeが(1, N)なので、sampled_actionも(1, N)になるため)
        action = tf.squeeze(sampled_action, axis=0)
        
        # アクションをクリッピング（ここは変更なし）
        action = tf.clip_by_value(action, -self.action_bound, self.action_bound)
        
        # バッチ次元が残っている場合があるので、numpyに変換してから返す
        # 元のコードの action.numpy()[0] とほぼ同じ挙動になります
        return action.numpy()

    def calculate_normal_dist_stats(self, mu, sigma, action):
        """
        tensorflow-probabilityなしで正規分布の統計量を計算する。

        Args:
            mu (tf.Tensor): 平均
            sigma (tf.Tensor): 標準偏差
            action (tf.Tensor): 確率を計算したいアクション

        Returns:
            tuple[tf.Tensor, tf.Tensor]: (log_prob, entropy)
        """
        # 数値安定性のための微小値
        epsilon = 1e-8
        stable_sigma = sigma + epsilon

        # 対数確率密度 (log_prob)
        log_prob = (
            -tf.math.log(stable_sigma)
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * tf.square((action - mu) / stable_sigma)
        )

        # エントロピー (entropy)
        entropy = 0.5 + 0.5 * np.log(2.0 * np.pi) + tf.math.log(stable_sigma)

        return log_prob, entropy


    def compute_loss(self, state, action, reward, next_state, done):
        state = tf.convert_to_tensor(state.reshape([1, self.state_dim]), dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state.reshape([1, self.state_dim]), dtype=tf.float32)
        action = tf.convert_to_tensor(action.reshape([1, self.action_dim]), dtype=tf.float32)
        reward = tf.cast(reward, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            mu, sigma, value = self.model(state)
            _, _, next_value = self.model(next_state)

            # Compute TD target and advantage
            target = reward + (1 - int(done)) * GAMMA * tf.squeeze(next_value)
            delta = target - tf.squeeze(value)

            # Critic loss (Mean Squared Error)
            critic_loss = delta ** 2

            # Actor loss
            #dist = tfp.distributions.Normal(mu, sigma)
            #log_prob = dist.log_prob(action)
            #entropy = dist.entropy()
            #actor_loss = -log_prob * delta - ENTROPY_COEF * entropy
            # ... (mu, sigma, action, delta, ENTROPY_COEFが定義されているとする)
            log_prob, entropy = self.calculate_normal_dist_stats(mu, sigma, action)
            actor_loss = -log_prob * delta - ENTROPY_COEF * entropy


        # Compute gradients
        actor_grads = tape.gradient(actor_loss, self.model.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.model.trainable_variables)
        return actor_loss, critic_loss, actor_grads, critic_grads

    def train(self):
        for ep in range(MAX_EPISODES):
            state, _ = self.env.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                action = self.get_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                actor_loss, critic_loss, actor_grads, critic_grads = self.compute_loss(
                    state, action, reward, next_state, done
                )

                # Apply gradients
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.model.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.model.trainable_variables))

                state = next_state
                episode_reward += reward
                if done or truncated:
                    break

            print(f"Episode: {ep + 1}, Reward: {episode_reward:.2f}")

if __name__ == "__main__":
    #import tensorflow_probability as tfp

    env = gym.make(ENV_NAME)
    agent = A2CAgent(env)
    agent.train()
