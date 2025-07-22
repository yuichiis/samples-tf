import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import time
import imageio

# === モデルの定義 ===
def create_a2c_model(input_shape, num_actions):
    input_layer = keras.layers.Input(shape=input_shape)
    common_layer = keras.layers.Dense(64, activation="relu",name="common_1")(input_layer)
    common_layer = keras.layers.Dense(64, activation="relu",name="common_2")(common_layer)
    last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
    actor_logits = keras.layers.Dense(num_actions, activation="linear", name="actor_logits", kernel_initializer=last_init)(common_layer)
    critic_value = keras.layers.Dense(1, activation="linear", name="critic_value")(common_layer)
    
    # ActorとCriticのモデルを分けることで、オプティマイザも分離できる
    actor = keras.Model(inputs=input_layer, outputs=actor_logits)
    critic = keras.Model(inputs=input_layer, outputs=critic_value)
    
    return actor, critic

# === 行動選択関数 ===
def get_action(actor_model, state):
    logits = actor_model(state.reshape((1, -1)))
    probs = tf.nn.softmax(logits)
    return np.random.choice(len(probs[0]), p=probs[0].numpy())

def get_best_action(actor_model, state):
    logits = actor_model(state.reshape((1, -1)))
    return tf.argmax(logits[0]).numpy()

# === GAEとリターンの計算関数 (PyTorch版をTFに移植) ★★★ ===
def compute_gae_and_returns(rewards, values, next_values, dones, gamma, lambda_gae):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        advantages[t] = delta + gamma * lambda_gae * last_advantage * mask # maskの位置を変更
        last_advantage = advantages[t] # maskの位置を変更
        
    returns = advantages + values
    return advantages, returns

# === 学習関数 (分離更新に対応) ★★★ ===
def update(actor, critic, actor_optimizer, critic_optimizer, experiences, gamma, lambda_gae, value_loss_weight, entropy_weight):
    states = np.asarray([e["state"] for e in experiences], dtype=np.float32)
    actions = np.asarray([e["action"] for e in experiences], dtype=np.int32)
    rewards = np.asarray([e["reward"] for e in experiences], dtype=np.float32)
    dones = np.asarray([e["done"] for e in experiences], dtype=np.bool_)
    
    # Criticを使ってV(s_t)とV(s_{t+1})を予測
    values = critic.predict(states, verbose=0).squeeze()
    next_states = np.asarray([e["n_state"] for e in experiences], dtype=np.float32)
    next_values = critic.predict(next_states, verbose=0).squeeze()
    
    # GAEとリターンを計算
    advantages, returns_to_go = compute_gae_and_returns(rewards, values, next_values, dones, gamma, lambda_gae)
    
    # アドバンテージの正規化
    advantages_norm = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    # --- Actorの学習 ---
    with tf.GradientTape() as tape:
        logits = actor(states, training=True)
        action_probs = tf.nn.softmax(logits)
        log_probs = tf.math.log(tf.clip_by_value(action_probs, 1e-10, 1.0))
        
        onehot_actions = tf.one_hot(actions, actor.output.shape[1])
        selected_log_probs = tf.reduce_sum(log_probs * onehot_actions, axis=1)
        
        entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * log_probs, axis=1))
        
        policy_loss = -tf.reduce_mean(selected_log_probs * advantages_norm) - entropy * entropy_weight

    actor_grads = tape.gradient(policy_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

    # --- Criticの学習 ---
    with tf.GradientTape() as tape:
        predicted_values = critic(states, training=True)

        #value_loss = tf.keras.losses.huber(returns_to_go, predicted_values)
        # まず、純粋なValue Lossを計算
        value_loss_pure = tf.keras.losses.huber(returns_to_go, predicted_values)
        
        # ★★★ 損失関数に係数を掛けてスケーリングする ★★★
        # これがPyTorchの (value_loss_coeff * value_loss).backward() と等価
        scaled_value_loss = value_loss_weight * value_loss_pure        

    #critic_grads = tape.gradient(value_loss, critic.trainable_variables)
    #critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
    #
    #return policy_loss.numpy(), tf.reduce_mean(value_loss).numpy(), entropy.numpy()

    # スケーリングされた損失の勾配を計算
    critic_grads = tape.gradient(scaled_value_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
    
    # ログには純粋な(スケーリング前の)損失を返す
    return policy_loss.numpy(), tf.reduce_mean(value_loss_pure).numpy(), entropy.numpy()

def evaluation(model):
    test_env = gym.make("CartPole-v1")
    all_rewards = []
    all_steps = []
    for i in range(10):
        state, _ = test_env.reset()
        done, truncated = False, False
        test_reward = 0
        test_step = 0
        while not (done or truncated):
            test_step += 1
            action = get_best_action(model, state)
            state, reward, done, truncated, _ = test_env.step(action)
            test_reward += reward
        all_rewards.append(test_reward)
        all_steps.append(test_step)
    test_env.close()
    val_reward = np.mean(all_rewards)
    val_step = np.mean(all_steps)
    return val_reward, val_step

def train(env,actor,critic):
    # --- ハイパーパラメータ (PyTorch版を参考に調整) ★★★ ---
    total_timesteps = 300000
    n_steps = 256 # 512 # バッチサイズを大きくすると安定しやすい
    gamma = 0.99
    lambda_gae = 0.95 # GAEのλ
    actor_lr = 3e-4
    critic_lr = 1e-3
    value_loss_weight = 0.5#0.25#0.5 # ★★★ この行を追加 ★★★
    entropy_weight = 0.01#0.02#0.01
    
    print('A2C GAE版')
    print('total_timesteps =',total_timesteps)
    print('n_steps =',n_steps)
    print('gamma =',gamma)
    print('lambda_gae =',lambda_gae)
    print('actor_lr =',actor_lr)
    print('critic_lr =',critic_lr)
    print('value_loss_weight =',value_loss_weight)
    print('entropy_weight =',entropy_weight)

    actor_optimizer = Adam(learning_rate=actor_lr)
    critic_optimizer = Adam(learning_rate=critic_lr)

    print("--- GAE-A2C 学習開始 ---")
    start_time = time.time()
    all_rewards = []
    all_p_losses = [] 
    all_v_losses = [] 
    all_entropies = []
    
    state, _ = env.reset()
    episode_reward_sum = 0
    episode_count = 0
    update_count = 0
    experiences = []

    for global_step in range(1, total_timesteps + 1):
        # --- N-stepのデータ収集 ---
        action = get_action(actor, state)
        n_state, reward, done, truncated, _ = env.step(action)
            
        experiences.append({
            "state": state, "action": action, "reward": reward,
            "n_state": n_state, "done": done
        })
            
        state = n_state
        episode_reward_sum += reward
            
        # --- 学習 ---
        if len(experiences) >= n_steps:
            p_loss, v_loss, entropy = update(actor, critic, actor_optimizer, critic_optimizer, experiences, gamma, lambda_gae, value_loss_weight, entropy_weight)
            all_p_losses.append(p_loss)
            all_v_losses.append(v_loss)
            all_entropies.append(entropy)
            experiences = []
            update_count += 1
        
        if done or truncated:
            all_rewards.append(episode_reward_sum)
            episode_count += 1
            episode_reward_sum = 0
            state, _ = env.reset()

        if global_step % (n_steps*10) == 0 or global_step == total_timesteps:
            avg_reward = np.mean(all_rewards[-20:]) if all_rewards else 0
            avg_p_loss = np.mean(all_p_losses[-20:]) if all_p_losses else 0
            avg_v_loss = np.mean(all_v_losses[-20:]) if all_v_losses else 0
            avg_entropy = np.mean(all_entropies[-20:]) if all_entropies else 0
            val_reward, val_step = evaluation(actor)
            print(f"Step:{global_step}/{total_timesteps//1000}k|Ep:{episode_count}|AvgRwd:{avg_reward:.1f}|P_Loss:{avg_p_loss:.3f}|V_Loss: {avg_v_loss:.3f}|Entropy:{avg_entropy:.3f}|ValSt:{val_step:.1f}|ValRwd:{val_reward:.1f}")
            if avg_reward > 475: # CartPole-v1のクリア基準
                print(f"Environment solved! Average reward: {avg_reward:.1f}")
                break


    print("--- 学習終了 ---")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"実行時間: {execution_time:.4f}秒")

    # ... (プロットとテストのコードは変更なし) ...
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(all_rewards, label='Episode Reward')
    if len(all_rewards) >= 20:
        moving_avg = np.convolve(all_rewards, np.ones(20)/20, mode='valid')
        plt.plot(np.arange(19, len(all_rewards)), moving_avg, label='Moving Average (20 ep)', color='orange')
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(all_p_losses, label='P-Loss', alpha=0.7)
    plt.plot(all_v_losses, label='V-Loss', alpha=0.7)
    plt.plot(all_entropies, label='Entropy', alpha=0.7)
    plt.title('Loss and Entropy per Update')
    plt.xlabel('Update Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === メイン処理 ===
if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    obs_shape = env.observation_space.shape
    nb_actions = env.action_space.n
    actor, critic = create_a2c_model(obs_shape, nb_actions)
    actor.summary()
    critic.summary()

    model_file = "cartpole-a2c-gae-nstep-{}.weights.h5"
    if os.path.isfile(model_file.format('actor')):
        actor.load_weights(model_file.format('actor'))
        critic.load_weights(model_file.format('critic'))
        #load_model(model_file)
    else:
        train(env,actor,critic)
        # 9. 学習済みモデルの保存と評価
        actor.save_weights(model_file.format('actor'))
        critic.save_weights(model_file.format('critic'))
    env.close()


    print("\n--- テスト実行 ---")
    env_render = gym.make("CartPole-v1", render_mode="rgb_array")
    for i in range(5):
        state, _ = env_render.reset()
        done, truncated = False, False
        test_reward = 0
        test_steps = 0
        frames = []
        while not (done or truncated):
            test_steps += 1
            frames.append(env_render.render())
            action = get_best_action(actor, state)
            state, reward, done, truncated, _ = env_render.step(action)
            test_reward += reward
        print(f"Test Episode {i+1}, Test Steps: {test_steps}, Total Reward: {test_reward}")
    env_render.close()
    imageio.mimsave('cartpole-a2c-gae-nstep.gif', frames, fps=30)
    print(f"GIFを'cartpole-a2c-gae-nstep.gif'に保存しました。最終報酬: {test_reward:.2f}")
    