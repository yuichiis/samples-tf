import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import gymnasium as gym

# TensorFlow 2.xではMish活性化関数が標準でサポートされています
# tf.keras.activations.mish

def t(x, dtype=tf.float32):
    """Numpy配列やリストをTensorFlowテンソルに変換します。"""
    return tf.convert_to_tensor(x, dtype=dtype)

class Actor(keras.Model):
    """アクターネットワーク：状態を入力とし、行動の確率分布を出力する"""
    def __init__(self, state_dim, n_actions, activation=tf.keras.activations.tanh):
        super().__init__()
        self.n_actions = n_actions
        self.model = keras.Sequential([
            layers.Input(shape=(state_dim,)),
            layers.Dense(64, activation=activation),
            layers.Dense(64, activation=activation),
            layers.Dense(n_actions)
        ])
        
        # logstdsを学習可能な変数として定義
        self.logstds = tf.Variable(tf.fill((n_actions,), 0.1), trainable=True, name="logstds")
    
    def call(self, X):
        """順伝播：状態を入力とし、行動の正規分布を返す"""
        means = self.model(X)
        stds = tf.clip_by_value(tf.exp(self.logstds), 1e-3, 50)
        
        # tensorflow_probabilityの分布を使用
        return tfp.distributions.Normal(loc=means, scale=stds)

class Critic(keras.Model):
    """クリティックネットワーク：状態を入力とし、その状態の価値を出力する"""
    def __init__(self, state_dim, activation=tf.keras.activations.tanh):
        super().__init__()
        self.model = keras.Sequential([
            layers.Input(shape=(state_dim,)),
            layers.Dense(64, activation=activation),
            layers.Dense(64, activation=activation),
            layers.Dense(1)
        ])
    
    def call(self, X):
        """順伝播：状態を入力とし、その状態の価値を返す"""
        return self.model(X)

def discounted_rewards(rewards, dones, gamma):
    """割引報酬和を計算します"""
    ret = 0
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + ret * gamma * (1 - done)
        discounted.append(ret)
    return discounted[::-1]

def process_memory(memory, n_actions, gamma=0.99, discount_rewards=True):
    """メモリ（経験のリスト）をTensorFlowテンソルのバッチに変換します"""
    actions, rewards, states, next_states, dones = [], [], [], [], []

    for action, reward, state, next_state, done in memory:
        actions.append(action)
        rewards.append(reward)
        states.append(state)
        next_states.append(next_state)
        dones.append(done)
    
    if discount_rewards:
        rewards = discounted_rewards(rewards, dones, gamma)

    actions = t(actions)
    states = t(states)
    next_states = t(next_states)
    rewards = tf.reshape(t(rewards), (-1, 1))
    dones = tf.reshape(t(dones, dtype=tf.float32), (-1, 1))
    return actions, rewards, states, next_states, dones

class A2CLearner:
    """A2Cアルゴリズムの学習ロジックを管理するクラス"""
    def __init__(self, actor, critic, gamma=0.9, entropy_beta=0,
                 actor_lr=4e-4, critic_lr=4e-3, max_grad_norm=0.5):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.actor = actor
        self.critic = critic
        self.entropy_beta = entropy_beta
        self.actor_optim = keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optim = keras.optimizers.Adam(learning_rate=critic_lr)

    def learn(self, memory, steps, discount_rewards=True):
        """メモリからバッチデータを作成し、ActorとCriticを更新する"""
        actions, rewards, states, next_states, dones = process_memory(
            memory, self.actor.n_actions, self.gamma, discount_rewards
        )

        # --- Criticの学習 ---
        with tf.GradientTape() as critic_tape:
            if discount_rewards:
                td_target = rewards
            else:
                # ターゲットの価値 V(s') は定数として扱う (勾配を計算しない)
                next_values_detached = tf.stop_gradient(self.critic(next_states))
                td_target = rewards + self.gamma * next_values_detached * (1 - dones)
            
            value = self.critic(states)
            critic_loss = tf.reduce_mean(tf.square(td_target - value))

        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.max_grad_norm)
        self.critic_optim.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # --- Actorの学習 ---
        with tf.GradientTape() as actor_tape:
            # Advantage (利得) を計算。この計算は勾配に影響させない
            advantage = tf.stop_gradient(td_target - value)

            norm_dists = self.actor(states)
            # actionが複数次元の場合、log_probを合計する
            logs_probs = tf.reduce_sum(norm_dists.log_prob(actions), axis=1, keepdims=True)
            entropy = tf.reduce_mean(norm_dists.entropy())
            
            actor_loss = tf.reduce_mean(-logs_probs * advantage) - entropy * self.entropy_beta

        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.max_grad_norm)
        self.actor_optim.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # --- TensorBoardへのログ記録 ---
        with writer.as_default(step=steps):
            tf.summary.scalar("losses/actor", actor_loss)
            tf.summary.scalar("losses/critic", critic_loss)
            tf.summary.scalar("losses/advantage", tf.reduce_mean(advantage))
            tf.summary.scalar("losses/log_probs", tf.reduce_mean(logs_probs))
            tf.summary.scalar("losses/entropy", entropy)
            tf.summary.histogram("gradients/actor", tf.concat([tf.reshape(g, [-1]) for g in actor_grads if g is not None], axis=0))
            tf.summary.histogram("gradients/critic", tf.concat([tf.reshape(g, [-1]) for g in critic_grads if g is not None], axis=0))
            tf.summary.histogram("parameters/actor", tf.concat([tf.reshape(p, [-1]) for p in self.actor.trainable_variables], axis=0))
            tf.summary.histogram("parameters/critic", tf.concat([tf.reshape(p, [-1]) for p in self.critic.trainable_variables], axis=0))

class Runner:
    """環境とエージェントの相互作用を実行し、経験を収集するクラス"""
    def __init__(self, env):
        self.env = env
        self.state = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.total_entropy = 0.0
        self.entropy_count = 0
    
    def reset(self):
        self.episode_reward = 0
        self.done = False
        self.state, _ = self.env.reset()
    
    def run(self, max_steps, memory=None):
        if not memory: memory = []
        
        for i in range(max_steps):
            if self.done: self.reset()
            
            # 状態をテンソルに変換し、バッチ次元を追加
            state_tensor = t(self.state[np.newaxis, :])
            dists = actor(state_tensor)
            actions = dists.sample().numpy()[0]  # バッチ次元を削除

            # 環境の行動範囲に合わせてクリップ
            actions_clipped = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            self.total_entropy += tf.reduce_mean(dists.entropy()).numpy()
            self.entropy_count += 1
            
            #print(
            #    "std:{:5.2f} action:{:5.2f}  clipped:{:5.2f}".format(
            #        tf.math.exp(actor.logstds).numpy()[0],
            #        actions[0],
            #        actions_clipped[0],
            #    )
            #)

            next_state, reward, terminated, truncated, info = self.env.step(actions_clipped)
            self.done = terminated or truncated
            memory.append((actions, reward, self.state, next_state, terminated))

            self.state = next_state
            self.steps += 1
            self.episode_reward += reward
            
            if self.done:
                self.episode_rewards.append(self.episode_reward)
                if len(self.episode_rewards) % 10 == 0:
                    last_std = tf.math.exp(actor.logstds).numpy()[0]
                    entropy = self.total_entropy / self.entropy_count
                    self.total_entropy = 0.0
                    self.entropy_count = 0
                    print(f"episode: {len(self.episode_rewards)}, episode reward: {self.episode_reward:.2f}, entropy: {entropy:.2f}, std: {last_std:.2f} ")
                
                with writer.as_default(step=self.steps):
                    tf.summary.scalar("episode_reward", self.episode_reward)
        return memory

# --- メイン実行ブロック ---
if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    writer = tf.summary.create_file_writer("runs_tf/mish_activation")

    # config
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    # Mish活性化関数を関数オブジェクトとして渡す
    actor = Actor(state_dim, n_actions, activation=tf.keras.activations.mish)
    critic = Critic(state_dim, activation=tf.keras.activations.mish)

    # モデルの重みを初期化するために、一度ダミーデータで呼び出す
    actor(np.random.rand(1, state_dim).astype(np.float32))
    critic(np.random.rand(1, state_dim).astype(np.float32))

    learner = A2CLearner(actor, critic)
    runner = Runner(env)

    # 学習パラメータ
    steps_on_memory = 16
    episodes = 500
    episode_length = 200
    total_learning_steps = (episode_length * episodes) // steps_on_memory

    print(f"Start training for {total_learning_steps} learning steps...")
    for i in range(total_learning_steps):
        memory = runner.run(steps_on_memory)
        # GAE(Generalized Advantage Estimation)ではなく、TD(0)で学習
        learner.learn(memory, runner.steps, discount_rewards=False)
        
        if (i + 1) % 100 == 0:
            print(f"Learning step: {i+1}/{total_learning_steps}")
            
    print("Training finished.")
    env.close()
    writer.close()