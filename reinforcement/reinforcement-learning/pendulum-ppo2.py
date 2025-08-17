import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # INFOとWARNINGログを抑制

import tensorflow as tf
import numpy as np
import gymnasium as gym
import imageio
# Pendulum (連続行動空間) のための正規分布を扱うためにtensorflow_probabilityをインポート
#import tensorflow_probability as tfp

# ===================================================================
# ヘルパー関数 (Pendulum用に不要になったものは削除・変更)
# ===================================================================

def standardize(
    x,         # (rolloutSteps)
    ddof=None,
    ) :
    # baseline
    mean = np.mean(x)     # ()

    baseX = x - mean                    # (rolloutSteps)
    # std
    if ddof:
        n = len(x)-1
    else :
        n = len(x)

    variance = np.sum(np.square(baseX)) / n                 # ()
    stdDev = np.sqrt(variance)                              # ()
    # standardize
    result = baseX / (stdDev + 1e-8)                        # (rolloutSteps)
    return result                                           # (rolloutSteps)

# === 正規分布関数 ===
def sample(mu_normalized,log_std):
    """tensorflow-probabilityなしで正規分布からサンプリング"""
    std = tf.exp(log_std)
    action_normalized = tf.random.normal(
        shape=tf.shape(mu_normalized), mean=mu_normalized, stddev=std
    )
    return action_normalized

def log_prob_entropy(mu, log_std, action):
    """
    tensorflow-probabilityなしで正規分布の統計量を計算する。
    Args:
        mu (tf.Tensor):      平均                       (batchsize,num_actions)
        log_std (tf.Tensor): 標準偏差のlog              (num_actions)
        action (tf.Tensor):  確率を計算したいアクション   (batchsize,num_actions)
    Returns:
        tuple[tf.Tensor, tf.Tensor]: (log_prob, entropy)
    """
    # 数値安定性のための微小値
    epsilon = 1e-8
    pi = tf.constant(np.pi, dtype=tf.float32)
    std = tf.math.exp(log_std)
    stable_std = std + epsilon
    # 対数確率密度 (log_prob)
    log_prob = (
        - tf.math.log(stable_std)
        - 0.5 * tf.math.log(2.0 * pi)
        - 0.5 * tf.square((action - mu) / stable_std)
    )
    # エントロピー (entropy)
    entropy = 0.5 + 0.5 * tf.math.log(2.0 * pi) + tf.math.log(stable_std)
    entropy = tf.zero_like(mu) + tf.expand_dims(entropy, axis=0)  # 他のテンソルとの互換性のため
    return log_prob, entropy  # (log_prob=(batchsize,num_actions), entropy=(batchsize,num_actions) )


# ===================================================================
# PPOの実装
# ===================================================================

# -------------------------------------------------------------------- #
# ハイパーパラメータをPendulum用に修正
# -------------------------------------------------------------------- #
ENV_NAME = 'Pendulum-v1'  # (変更) 環境名をPendulumに変更
GAMMA = 0.9 # 0.99
GAE_LAMBDA = 0.95
LEARNING_RATE = 3e-4 #1e-3
CLIP_EPSILON = 0.2
N_EPOCHS = 10
BATCH_SIZE = 64
N_ROLLOUT_STEPS = 1024 # 2048
TARGET_SCORE = -250 # (変更) Pendulumの目標スコアに変更 (報酬は最大で0)
VALUE_LOSS_WEIGHT = 0.5
ENTROPY_WEIGHT = 0.01 # rl_zoo3では0.0
STANDARDIZE = True


# (変更) Actor-Criticモデルを連続行動空間用に改造
class ActorCritic(tf.keras.Model):
    """Actor-Criticモデル (連続行動空間対応)"""
    def __init__(self, action_dim):
        super().__init__()
        # 共通層はそのまま
        self.common_layer1 = tf.keras.layers.Dense(128, activation='relu', name='layer1')
        self.common_layer2 = tf.keras.layers.Dense(128, activation='relu', name='layer2')
        
        # (変更) Actorヘッド: 行動の平均(mu)を出力
        self.actor_mu = tf.keras.layers.Dense(action_dim, name='action_head')
        
        # (追加) Actorの標準偏差(std): 学習可能な変数としてlog_stdを定義
        # 状態に依存しない固定の標準偏差を学習させるアプローチ
        #self.log_std = tf.Variable(tf.zeros(action_dim), trainable=True, name='log_std')
        self.log_std = self.add_weight(
            name='log_std',
            shape=(action_dim,),
            initializer='zeros',
            trainable=True
        )
        
        # Criticヘッドはそのまま
        self.critic_head = tf.keras.layers.Dense(1, name='critic_head')

    def call(self, state):
        x = self.common_layer1(state)
        x = self.common_layer2(x)
        
        # (変更) 平均(mu)と状態価値(value)を返す
        mu = self.actor_mu(x)
        value = self.critic_head(x)
        return mu, value

def compute_advantages_and_returns(rewards, values, dones):
    """GAEとリターンを計算する (変更なし)"""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0

    for t in reversed(range(len(rewards))):
        if dones[t]:
            delta = rewards[t] - values[t]
            last_advantage = delta
        else:
            delta = rewards[t] + GAMMA * values[t+1] - values[t]
            last_advantage = delta + GAMMA * GAE_LAMBDA * last_advantage
        advantages[t] = last_advantage
        
    returns = advantages + values[:-1]
    
    return advantages, returns


# (変更) 決定論的な最善行動を取得する関数 (分布の平均値を使う)
def get_best_action(model, state, action_bound):
    mu, _ = model(state.reshape((1, -1)))
    # 平均値をそのまま行動とし、環境の行動範囲にクリップする
    action = np.clip(mu.numpy().flatten(), -action_bound, action_bound)
    return action

def main():
    # (変更) Pendulum環境のセットアップ
    env = gym.make(ENV_NAME)
    # 行動の次元数と範囲を取得
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    obs_shape = env.observation_space.shape

    model = ActorCritic(action_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    #model(tf.zeros([1]+list(obs_shape)))
    #print('num of trainable_variables=',len(model.trainable_variables))
    #print(model.trainable_variables)

    episode_count = 0
    total_step = 0
    episode_score = 0
    last_episode_scores = []
    
    while True:
        episode_count += 1
        avg_loss = 0
        avg_a_loss = 0
        avg_c_loss = 0
        avg_entropy = 0
        
        # === 1. データ収集 (Rollout) (連続行動用に変更) ===
        # (変更) 古い対数確率(log_probs)も保存するメモリを追加
        states_mem, actions_mem, rewards_mem, dones_mem, log_probs_mem = [], [], [], [], []
        state, _ = env.reset()
        
        for t in range(N_ROLLOUT_STEPS):
            total_step += 1
            state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
            
            # (変更) モデルからmuを取得し、正規分布を定義
            mu, _ = model(state_tensor)
            #std = tf.exp(model.log_std)
            #dist = tfp.distributions.Normal(loc=mu, scale=std)
            
            # (変更) 分布から行動をサンプリング
            action = sample(mu,model.log_std)
            # (変更) サンプリングした行動の対数確率を計算
            log_prob, _ = log_prob_entropy(mu, model.log_std, action)
            
            # (変更) 行動を環境の範囲内にクリップして実行
            clipped_action = np.clip(action.numpy().flatten(), -action_bound, action_bound)
            next_state, reward, done, truncated, _ = env.step(clipped_action)
            episode_score += reward
            
            # (変更) メモリに保存 (サンプリングした元のactionとlog_probを保存)
            states_mem.append(state)
            actions_mem.append(action.numpy().flatten())
            rewards_mem.append(reward)
            dones_mem.append(done or truncated)
            log_probs_mem.append(log_prob.numpy().flatten())
            
            state = next_state
            if done or truncated:
                state, _ = env.reset()
                # Pendulumは常に報酬がマイナスなので、スコアが溜まっていく
                if len(last_episode_scores) > 100:
                    last_episode_scores.pop(0)
                last_episode_scores.append(episode_score)
                episode_score = 0

        # === 2. 学習データの準備 (変更あり) ===
        states_tensor = tf.convert_to_tensor(np.array(states_mem), dtype=tf.float32)
        # (変更) actions_tensorのdtypeをfloat32に
        actions_tensor = tf.convert_to_tensor(np.array(actions_mem), dtype=tf.float32)
        # (変更) 保存しておいた古い対数確率を使用
        old_log_probs_tensor = tf.convert_to_tensor(np.array(log_probs_mem), dtype=tf.float32)

        # GAE計算のための価値関数値を取得 (元のコードのロジックを維持)
        _, values_old_tensor = model(states_tensor)
        next_state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        _, next_value_tensor = model(next_state_tensor)
        
        values_for_gae = np.append(tf.squeeze(values_old_tensor).numpy(), tf.squeeze(next_value_tensor).numpy())
        
        advantages, returns = compute_advantages_and_returns(
            rewards_mem, values_for_gae, dones_mem
        )

        if STANDARDIZE:
            advantages = standardize(advantages)

        advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices((
            states_tensor, actions_tensor, old_log_probs_tensor, advantages_tensor, returns_tensor
        ))
        dataset = dataset.shuffle(buffer_size=N_ROLLOUT_STEPS).batch(BATCH_SIZE)

        # === 3. モデルの学習 (連続行動用に変更) ===
        num_batches = 0
        for _ in range(N_EPOCHS):
            for batch in dataset:
                states_b, actions_b, old_log_probs_b, advantages_b, returns_b = batch
                with tf.GradientTape() as tape:
                    # (変更) 現在のモデルで新しい分布を予測
                    new_mu, new_values = model(states_b)
                    new_values = tf.squeeze(new_values)
                    #new_std = tf.exp(model.log_std)
                    #new_dist = tfp.distributions.Normal(loc=new_mu, scale=new_std)
                    
                    # (変更) 新しい分布での対数確率とエントロピーを計算
                    # new_mu        = (batchsize,num_action)
                    # new_std       = (num_action)
                    # actions_b     = (batchsize,num_action)
                    # new_log_probs = (batchsize,num_action)
                    # entropy       = (batchsize,num_action)
                    #new_log_probs = new_dist.log_prob(actions_b)
                    #entropy = new_dist.entropy()

                    # new_mu        = (batchsize,num_action)
                    # new_std       = (num_action)
                    # actions_b     = (batchsize,num_action)
                    # new_log_probs = (batchsize,num_action)
                    # entropy       = (batchsize,num_action)
                    new_log_probs, entropy = log_prob_entropy(new_mu, model.log_std, actions_b)
                    #print('new_mu=',new_mu.shape)
                    #print('log_std=',model.log_std.shape)
                    #print('actions_b=',actions_b.shape)
                    #print('new_log_probs=',new_log_probs.shape)
                    #print('entropy=',entropy.shape)
                    
                    # 多次元行動も考慮し、log_probとentropyをスカラーに変換
                    new_log_probs = tf.reduce_sum(new_log_probs, axis=1)        # (batchsize,1)
                    old_log_probs_b = tf.reduce_sum(old_log_probs_b, axis=1)    # (batchsize,1)
                    entropy = tf.reduce_sum(entropy, axis=1)                    # (batchsize,1)
                    #print('new_log_probs=',new_log_probs.shape)
                    #print('old_log_probs_b=',old_log_probs_b.shape)
                    #print('entropy=',entropy.shape)

                    # 1. Actor Loss (Policy Loss) - 計算式自体は同じ
                    ratio = tf.exp(new_log_probs - old_log_probs_b)
                    clipped_ratio = tf.clip_by_value(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
                    actor_loss = -tf.reduce_mean(
                        tf.minimum(ratio * advantages_b, clipped_ratio * advantages_b)
                    )
                    
                    # 2. Critic Loss (Value Loss) - 同じ
                    critic_loss = tf.reduce_mean(tf.square(returns_b - new_values))
                    
                    # 3. Entropy Loss - 同じ
                    entropy_loss = -tf.reduce_mean(entropy)

                    #print('actor_loss=',actor_loss.numpy())
                    #print('critic_loss=',critic_loss.numpy())
                    #print('entropy_loss=',entropy_loss.numpy())
                    #print(
                    #    "actor_loss={:+3.3f},critic_loss={:7.1f},entropy_loss={:+1.3f}".format(
                    #    actor_loss.numpy(),critic_loss.numpy(),entropy_loss.numpy()
                    #    )
                    #)

                    total_loss = actor_loss + VALUE_LOSS_WEIGHT * critic_loss + ENTROPY_WEIGHT * entropy_loss
                    
                grads = tape.gradient(total_loss, model.trainable_variables)
                #print('num_grads=',len(grads))
                grads, _ = tf.clip_by_global_norm(grads, 0.5)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                avg_loss += total_loss
                avg_a_loss += actor_loss
                avg_c_loss += critic_loss
                avg_entropy += -entropy_loss
                num_batches += 1

        avg_loss /= num_batches
        avg_a_loss /= num_batches
        avg_c_loss /= num_batches
        avg_entropy /= num_batches

        # === 4. 進捗の評価と表示 (変更あり) ===
        eval_env = gym.make(ENV_NAME)
        eval_scores = []
        for _ in range(10):
            state, _ = eval_env.reset()
            done, truncated = False, False
            score = 0
            while not (done or truncated):
                # (変更) 評価時は決定論的な行動を選択
                action = get_best_action(model, state, action_bound)
                state, reward, done, truncated, _ = eval_env.step(action)
                score += reward
            eval_scores.append(score)
        avg_evl_score = np.mean(eval_scores)
        
        avg_score = np.mean(last_episode_scores) if last_episode_scores else -1600 # 初期値
        
        print(f"Ep:{episode_count}, St:{total_step}, Scr:{avg_score:.1f}, ALoss:{avg_a_loss:.3f}, CLoss:{avg_c_loss:.3f}, Ety:{avg_entropy:.4f}, EvScr:{avg_evl_score:.1f}")

        if avg_score >= TARGET_SCORE:
            print(f"\n目標スコア {TARGET_SCORE} を達成しました！学習を終了します。")
            break

    print("\n--- テスト実行 ---")
    # (変更) レンダリング用の環境もPendulumに変更
    env_render = gym.make("Pendulum-v1", render_mode="rgb_array")
    frames = []
    for i in range(3): # GIFが長くなりすぎないように3回に
        state, _ = env_render.reset()
        done, truncated = False, False
        test_reward = 0
        
        while not (done or truncated):
            frames.append(env_render.render())
            # (変更) get_best_actionを使用
            action = get_best_action(model, state, action_bound)
            state, reward, done, truncated, _ = env_render.step(action)
            test_reward += reward
        print(f"Test Episode {i+1}, Total Reward: {test_reward:.2f}")
    
    env_render.close()
    
    gif_path = 'pendulum-ppo2.gif'
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"GIFを'{gif_path}'に保存しました。")


if __name__ == '__main__':
    # 依存ライブラリのチェック
    try:
        import tensorflow_probability
    except ImportError:
        print("エラー: tensorflow_probability がインストールされていません。")
        print("次のコマンドでインストールしてください: pip install tensorflow-probability")
        exit()
    main()
    