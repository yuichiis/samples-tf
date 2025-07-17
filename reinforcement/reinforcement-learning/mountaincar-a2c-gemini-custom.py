import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import collections

# ハイパーパラメータ
GAMMA = 0.99
# 学習が難しいタスクなので、少し学習率を下げて安定させる
LEARNING_RATE = 0.0005 
MAX_EPISODES = 2000 # 念のためエピソード数を増やす
ENTROPY_BETA = 0.01

class ActorCritic(Model):
    """ActorとCriticを統合したモデル"""
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # ネットワークを少し大きくして表現力を上げる
        self.common_dense1 = Dense(256, activation='relu', name='common_dense1')
        self.common_dense2 = Dense(256, activation='relu', name='common_dense2')
        
        self.actor_output = Dense(action_dim, activation='linear', name='actor_output')
        self.critic_output = Dense(1, activation='linear', name='critic_output')

    def call(self, inputs):
        """モデルの順伝播"""
        x = self.common_dense1(inputs)
        x = self.common_dense2(x)
        actor_logits = self.actor_output(x)
        critic_value = self.critic_output(x)
        return actor_logits, critic_value

def train():
    env = gym.make('MountainCar-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim)
    optimizer = Adam(learning_rate=LEARNING_RATE)

    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    ep_rewards = collections.deque(maxlen=100)

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_dim])
        
        episode_reward = 0
        
        # ★★★ カスタム報酬のための変数を初期化 ★★★
        # エピソード内の最高到達地点を記録 (車の位置の初期値は-0.5あたり)
        max_pos_in_episode = -1.0 
        
        with tf.GradientTape() as tape:
            states_hist = []
            actions_hist = []
            rewards_hist = []
            
            while True:
                state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
                actor_logits, critic_value = model(state_tensor)
                action_probs = tf.nn.softmax(actor_logits)
                action = np.random.choice(action_dim, p=np.squeeze(action_probs))
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # ★★★ ここからカスタム報酬の計算 ★★★
                car_position = next_state[0]
                
                # 1. 最高地点を更新したらボーナスを与える
                if car_position > max_pos_in_episode:
                    # 更新した差分に応じて報酬を追加
                    reward += (car_position - max_pos_in_episode) * 10 
                    max_pos_in_episode = car_position

                # 2. ゴールに到達したら大きなボーナスを与える
                if terminated:
                    reward += 100
                # ★★★ カスタム報酬ここまで ★★★

                next_state = np.reshape(next_state, [1, state_dim])
                
                states_hist.append(state_tensor)
                actions_hist.append(action)
                rewards_hist.append(reward)
                
                state = next_state
                episode_reward += reward # オリジナルの報酬ではなくカスタム後の報酬を記録
                
                if done:
                    break

            # (学習フェーズは変更なし)
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
        
        # 表示する報酬は、比較のためにオリジナルの報酬に戻すこともできる
        # ここではカスタム後の報酬で評価
        ep_rewards.append(episode_reward)
        avg_reward = np.mean(ep_rewards)

        print(f"Episode: {episode}, Custom Reward: {episode_reward:.2f}, Avg Custom Reward (last 100): {avg_reward:.2f}")

        # 解決条件もカスタム報酬ベースに変更（値は調整が必要）
        if avg_reward >= 50.0 and len(ep_rewards) >= 100:
            print(f"\nSolved at episode {episode}!")
            model.save_weights('mountaincar-a2c-custom-reward.weights.h5')
            break
            
    env.close()

if __name__ == '__main__':
    train()