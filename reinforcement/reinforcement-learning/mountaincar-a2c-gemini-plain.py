import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import collections

# ハイパーパラメータ
GAMMA = 0.99  # 割引率
LEARNING_RATE = 0.001 # 学習率
MAX_EPISODES = 1000   # 最大エピソード数
ENTROPY_BETA = 0.01   # エントロピー項の係数（探索を促進）

class ActorCritic(Model):
    """ActorとCriticを統合したモデル"""
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 共通の隠れ層
        self.common_dense1 = Dense(128, activation='relu', name='common_dense1')
        
        # Actor（方策）の出力層
        self.actor_output = Dense(action_dim, activation='linear', name='actor_output')
        
        # Critic（価値）の出力層
        self.critic_output = Dense(1, activation='linear', name='critic_output')

    def call(self, inputs):
        """モデルの順伝播"""
        x = self.common_dense1(inputs)
        actor_logits = self.actor_output(x)
        critic_value = self.critic_output(x)
        return actor_logits, critic_value

def train():
    # 環境の初期化
    env = gym.make('MountainCar-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # モデルとオプティマイザの初期化
    model = ActorCritic(state_dim, action_dim)
    optimizer = Adam(learning_rate=LEARNING_RATE)

    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    ep_rewards = collections.deque(maxlen=100)

    for episode in range(MAX_EPISODES):
        # 修正点 1: env.reset()の戻り値のタプルをアンパックする
        state, _ = env.reset()
        state = np.reshape(state, [1, state_dim])
        
        episode_reward = 0
        
        with tf.GradientTape() as tape:
            states_hist = []
            actions_hist = []
            rewards_hist = []
            
            # --- 1エピソード実行フェーズ ---
            while True:
                state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
                actor_logits, critic_value = model(state_tensor)
                action_probs = tf.nn.softmax(actor_logits)
                action = np.random.choice(action_dim, p=np.squeeze(action_probs))
                
                # 修正点 2: env.step()の戻り値を新しいAPIに合わせる
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated # 終了フラグを計算

                next_state = np.reshape(next_state, [1, state_dim])
                
                states_hist.append(state_tensor)
                actions_hist.append(action)
                rewards_hist.append(reward)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break

            # --- 学習フェーズ ---
            discounted_rewards = []
            cumulative_reward = 0
            for r in rewards_hist[::-1]:
                cumulative_reward = r + GAMMA * cumulative_reward
                discounted_rewards.insert(0, cumulative_reward)
                
            states_tensor = tf.concat(states_hist, axis=0)
            actions_tensor = tf.convert_to_tensor(actions_hist, dtype=tf.int32)
            discounted_rewards_tensor = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)

            actor_logits_batch, critic_value_batch = model(states_tensor)
            
            advantage = discounted_rewards_tensor - tf.squeeze(critic_value_batch)
            
            actor_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=actions_tensor, logits=actor_logits_batch)
            actor_loss *= tf.stop_gradient(advantage)

            critic_loss = huber_loss(tf.expand_dims(discounted_rewards_tensor, 1), critic_value_batch)
            
            action_probs_batch = tf.nn.softmax(actor_logits_batch)
            entropy_loss = -tf.reduce_sum(action_probs_batch * tf.math.log(action_probs_batch + 1e-10), axis=1)
            entropy_loss *= ENTROPY_BETA
            
            total_loss = tf.reduce_sum(actor_loss) + critic_loss - tf.reduce_sum(entropy_loss)

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        ep_rewards.append(episode_reward)
        avg_reward = np.mean(ep_rewards)

        print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Avg Reward (last 100): {avg_reward:.2f}")

        if avg_reward >= -110.0 and len(ep_rewards) >= 100:
            print(f"\nSolved at episode {episode}!")
            model.save_weights('mountaincar-a2c.weights.h5')
            break
            
    env.close()

if __name__ == '__main__':
    train()