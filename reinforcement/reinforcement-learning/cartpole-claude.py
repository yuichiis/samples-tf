import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym
import random
from collections import deque

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# DQN (Deep Q-Network) agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay memory
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration rate decay
        self.learning_rate = 0.001  # Learning rate
        self.model = self._build_model()  # Build model
        self.target_model = self._build_model()  # Target model
        self.update_target_model()  # Initialize target model

    def _build_model(self):
        # Build neural network model
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Update target model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Select action using ε-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Select random action
        
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # Select action with highest Q-value

    def replay(self, batch_size):
        # Sample minibatch from replay memory and train
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        # Load model weights
        self.model.load_weights(name)

    def save(self, name):
        # Save model weights
        self.model.save_weights(name)

# Main function
def main():
    # Create CartPole environment
    env = gym.make('CartPole-v1')
    env.reset(seed=42)
    
    # Get state and action sizes
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize DQN agent
    agent = DQNAgent(state_size, action_size)
    
    # Training parameters
    episodes = 500
    batch_size = 32
    
    # Record rewards for each episode
    scores = []
    
    for e in range(episodes):
        # Reset environment
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        
        # Run one episode
        total_reward = 0
        for time in range(500):  # Maximum 500 steps
            # Select action
            action = agent.act(state)
            
            # Execute selected action
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            # Set reward (negative reward on game over)
            reward = reward if not done or time == 499 else -10
            
            # Store experience in memory
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Accumulate reward
            total_reward += reward
            
            # Handle end of episode
            if done:
                # Update target network
                agent.update_target_model()
                print(f"Episode: {e+1}/{episodes}, Score: {time+1}, Epsilon: {agent.epsilon:.2f}")
                break
        
        # Record episode reward
        scores.append(total_reward)
        
        # Train with minibatch from replay memory
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    
    # Save model
    agent.save("cartpole_dqn.h5")
    
    # Plot learning curve
    import matplotlib.pyplot as plt
    plt.plot(scores)
    plt.title('Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('learning_curve.png')
    plt.show()

if __name__ == "__main__":
    main()