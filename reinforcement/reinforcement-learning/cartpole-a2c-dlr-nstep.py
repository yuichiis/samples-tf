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
def get_action(model, state):
    logits = model(state.reshape((1, -1)))
    # probs = tf.nn.softmax(logits)
    # return np.random.choice(len(probs[0]), p=probs[0].numpy())
    action_tensor = tf.random.categorical(logits, 1)[0, 0]
    return action_tensor.numpy()

def get_best_action(model, state):
    logits = model(state.reshape((1, -1)))
    return tf.argmax(logits[0]).numpy()

# === 学習関数 (修正済み) ===
def update(actor, critic, actor_optimizer, critic_optimizer, experiences, gamma, value_loss_weight, entropy_weight, standardize):
    states = np.asarray([e["state"] for e in experiences], dtype=np.float32)
    actions = np.asarray([e["action"] for e in experiences], dtype=np.int32)
    rewards = np.asarray([e["reward"] for e in experiences], dtype=np.float32)
    dones = np.asarray([e["done"] for e in experiences], dtype=np.bool_)

    if not dones[-1]:
        last_state = experiences[-1]["n_state"].reshape((1, -1))
        last_v = critic(last_state)
        last_v = last_v.numpy()[0, 0]
    else:
        last_v = 0.0

    discounted_rewards = []
    G = last_v
    for r, done in zip(rewards[::-1], dones[::-1]):
        if done:
            G = 0
        G = r + gamma * G
        discounted_rewards.append(G)
    discounted_rewards.reverse()
    discounted_rewards = np.array(discounted_rewards, dtype=np.float32).reshape(-1, 1)

    # # ベースライン処理
    # discounted_rewards -= np.mean(discounted_rewards)  # 報酬の平均を引く

    onehot_actions = tf.one_hot(actions, actor.output[0].shape[0])
    # ★★★ 修正点 1: Advantage計算時にCriticの勾配を止める ★★★
    # これでAdvantageの計算がCriticの学習に影響を与えないことが保証される
    next_values = critic(states, training=False)

    with tf.GradientTape() as tape:
        logits = actor(states, training=True)

        advantage = discounted_rewards - next_values
        
        if(standardize):
            advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)

        action_probs = tf.nn.softmax(logits)
        selected_action_probs = tf.reduce_sum(action_probs * onehot_actions, axis=1, keepdims=True)
        selected_action_probs = tf.clip_by_value(selected_action_probs, 1e-10, 1.0)
        
        action_probs_clipped = tf.clip_by_value(action_probs, 1e-10, 1.0)
        entropy = -tf.reduce_sum(action_probs_clipped * tf.math.log(action_probs_clipped), axis=1, keepdims=True)

        # Policy Lossは正規化されたAdvantageを使う
        # tf.stop_gradientは不要になったが、念のため残しても害はない
        policy_loss = -tf.reduce_mean(tf.math.log(selected_action_probs) * advantage) - entropy_weight * entropy

    gradients = tape.gradient(policy_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(gradients, actor.trainable_variables))

    with tf.GradientTape() as tape:
        # Critic (Value) Lossは、Actorの学習とは独立して、vがG_tに近づくように学習する
        v = critic(states, training=True)
        value_loss_pure = tf.keras.losses.huber(discounted_rewards, v)

        scaled_value_loss = value_loss_weight * value_loss_pure
        scaled_value_loss = tf.reduce_mean(scaled_value_loss)

    gradients = tape.gradient(scaled_value_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
    
    return tf.reduce_mean(policy_loss).numpy(),tf.reduce_mean(value_loss_pure).numpy(), tf.reduce_mean(entropy).numpy()

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
    # --- ハイパーパラメータ (修正済み) ---
    standardize = True
    total_timesteps = 300000
    n_steps = 256#512#32
    gamma = 0.99
    actor_lr = 3e-4
    critic_lr = 1e-3
    #clipnorm = 0.5
    value_loss_weight = 0.5
    entropy_weight = 0.01

    print('A2C Different Learning Rates版')
    print('standardize =',standardize)    
    print('total_timesteps =',total_timesteps)
    print('n_steps =',n_steps)
    print('gamma =',gamma)
    print('actor_lr =',actor_lr)
    print('critic_lr =',critic_lr)
    #print('clipnorm =',clipnorm)
    print('value_loss_weight =',value_loss_weight)
    print('entropy_weight =',entropy_weight)

    actor_optimizer = Adam(learning_rate=actor_lr) #, clipnorm=clipnorm)
    critic_optimizer = Adam(learning_rate=critic_lr) #, clipnorm=clipnorm)

    print("--- 学習開始 ---")
    start_time = time.time()
    all_rewards = []
    all_episode_steps = []
    all_p_losses = [] 
    all_v_losses = [] 
    all_entropies = []
    
    experiences = []
    episode_count = 0
    update_count = 0
    episode_reward_sum = 0
    episode_step_count = 0
    
    state, _ = env.reset()

    for global_step in range(1, total_timesteps + 1):
        episode_step_count += 1
        action = get_action(actor, state)
        n_state, reward, done, truncated, _ = env.step(action)
        
        episode_reward_sum += reward

        experiences.append({
            "state": state, "action": action, "reward": reward,
            "n_state": n_state, "done": done
        })
        
        state = n_state

        if done or truncated:
            all_rewards.append(episode_reward_sum)
            all_episode_steps.append(episode_step_count)
            episode_count += 1
            episode_reward_sum = 0
            episode_step_count = 0
            state, _ = env.reset()

        if len(experiences) >= n_steps:
            p_loss, v_loss, entropy = update(actor,critic, actor_optimizer, critic_optimizer, experiences, gamma, value_loss_weight, entropy_weight, standardize)
            all_p_losses.append(p_loss)
            all_v_losses.append(v_loss)
            all_entropies.append(entropy)
            experiences = []
            update_count += 1

        if (global_step % (n_steps*10) == 0) or global_step == total_timesteps:
            avg_reward = np.mean(all_rewards[-20:])
            avg_episode_steps = np.mean(all_episode_steps[-20:])
            last_p_loss = all_p_losses[-1] if all_p_losses else 0
            last_v_loss = all_v_losses[-1] if all_v_losses else 0
            last_entropy = all_entropies[-1] if all_entropies else 0
            val_reward, val_step = evaluation(actor)
            print(f'St:{global_step}/{total_timesteps//1000}k|Ep:{episode_count}|AvgSt:{avg_episode_steps:.1f}|AvgRwd:{avg_reward:.1f}|P_Loss:{last_p_loss:.3f}|V_Loss:{last_v_loss:.3f}|Entropy:{last_entropy:.3f}|ValSt:{val_step:.1f}|ValRwd:{val_reward:.1f}')
            if avg_reward > 475:
                print(f"環境がクリアされました！ (平均報酬: {avg_reward})")
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
    plt.plot(all_p_losses, label='P_Loss', alpha=0.7)
    plt.plot(all_v_losses, label='V_Loss', alpha=0.7)
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
    actor,critic = create_a2c_model(obs_shape, nb_actions)
    actor.summary()
    critic.summary()

    model_file = "cartpole-a2c-dlr-nstep-{}.weights.h5"
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
    imageio.mimsave('cartpole-a2c-dlr-nstep.gif', frames, fps=30)
    print(f"GIFを'cartpole-dlr-a2c-nstep.gif'に保存しました。最終報酬: {test_reward:.2f}")
