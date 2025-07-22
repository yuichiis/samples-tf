import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque
import random

# ハイパーパラメータ
LEARNING_RATE_ACTOR = 0.001
LEARNING_RATE_CRITIC = 0.002
GAMMA = 0.99
ENTROPY_BETA = 0.01
MAX_EPISODES = 1000
MAX_STEPS = 200

class A2CAgent:
    def __init__(self, state_size, action_size, action_bound):
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        
        # Actor Network
        self.actor = self._build_actor()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_ACTOR)
        
        # Critic Network
        self.critic = self._build_critic()
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_CRITIC)
        
    def _build_actor(self):
        """Actor Network - 連続動作空間用"""
        inputs = tf.keras.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        
        # 平均値（mu）と標準偏差（sigma）を出力
        mu = tf.keras.layers.Dense(self.action_size, activation='tanh')(x)
        mu = tf.keras.layers.Lambda(lambda x: x * self.action_bound)(mu)
        
        sigma = tf.keras.layers.Dense(self.action_size, activation='softplus')(x)
        sigma = tf.keras.layers.Lambda(lambda x: x + 1e-5)(sigma)  # 数値安定性のため
        
        model = tf.keras.Model(inputs=inputs, outputs=[mu, sigma])
        return model
    
    def _build_critic(self):
        """Critic Network - 状態価値関数"""
        inputs = tf.keras.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        value = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=value)
        return model
    
    def get_action(self, state):
        """行動を選択"""
        state = tf.expand_dims(state, axis=0)
        mu, sigma = self.actor(state)
        
        # 正規分布からサンプリング
        dist = tf.random.normal(shape=mu.shape, mean=mu, stddev=sigma)
        action = tf.clip_by_value(dist, -self.action_bound, self.action_bound)
        
        return action.numpy()[0], mu.numpy()[0], sigma.numpy()[0]
    
    def train(self, states, actions, rewards, next_states, dones):
        """A2Cの学習"""
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Advantageの計算
        with tf.GradientTape() as tape:
            values = tf.squeeze(self.critic(states))
            next_values = tf.squeeze(self.critic(next_states))
            
            # TD Target
            td_targets = rewards + GAMMA * next_values * (1 - dones)
            advantages = td_targets - values
            
            # Critic Loss
            critic_loss = tf.reduce_mean(tf.square(advantages))
        
        # Criticの更新
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # Actor Loss
        with tf.GradientTape() as tape:
            mu, sigma = self.actor(states)
            
            # ログ確率の計算
            log_probs = -0.5 * tf.reduce_sum(
                tf.square((actions - mu) / sigma) + 2 * tf.math.log(sigma) + tf.math.log(2 * np.pi),
                axis=1
            )
            
            # エントロピーボーナス
            entropy = tf.reduce_mean(tf.reduce_sum(tf.math.log(sigma), axis=1))
            
            # Actor Loss = -log_prob * advantage - entropy_bonus
            actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages)) - ENTROPY_BETA * entropy
        
        # Actorの更新
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        return critic_loss.numpy(), actor_loss.numpy()

def train_a2c():
    """A2Cの学習メイン関数"""
    env = gym.make('Pendulum-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    
    agent = A2CAgent(state_size, action_size, action_bound)
    
    # 学習記録
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    
    for episode in range(MAX_EPISODES):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        episode_reward = 0
        episode_states = []
        episode_actions = []
        episode_rewards_step = []
        episode_next_states = []
        episode_dones = []
        
        for step in range(MAX_STEPS):
            action, mu, sigma = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards_step.append(reward)
            episode_next_states.append(next_state)
            episode_dones.append(done or truncated)
            
            state = next_state
            episode_reward += reward
            
            if done or truncated:
                break
        
        # エピソード終了時に学習
        if len(episode_states) > 0:
            critic_loss, actor_loss = agent.train(
                episode_states, episode_actions, episode_rewards_step,
                episode_next_states, episode_dones
            )
            
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
        
        episode_rewards.append(episode_reward)
        
        # 進捗表示
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    env.close()
    
    # 結果の可視化
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(actor_losses)
    plt.title('Actor Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(critic_losses)
    plt.title('Critic Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()
    
    return agent

def test_agent(agent, num_episodes=5):
    """学習済みエージェントのテスト"""
    env = gym.make('Pendulum-v1', render_mode='human')
    
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            action, _, _ = agent.get_action(state)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                break
        
        print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    print("Starting A2C training on Pendulum-v1...")
    
    # 学習実行
    trained_agent = train_a2c()
    
    # 学習済みエージェントのテスト
    print("\nTesting trained agent...")
    test_agent(trained_agent)
    
    print("Training completed!")
    