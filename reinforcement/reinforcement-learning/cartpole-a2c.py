import numpy as np
from collections import deque
from tensorflow.python import keras
from tensorflow.keras.optimizers import Adam
import gymnasium as gym
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

# 環境の状態の形式(shape)
obs_shape = env.observation_space.shape

# 環境の取りうるアクション数
nb_actions = env.action_space.n

lr = 0.01  # 学習率

c = input_ = keras.layers.Input(shape=obs_shape)
c = keras.layers.Dense(10, activation="relu")(c)
c = keras.layers.Dense(10, activation="relu")(c)
actor_layer = keras.layers.Dense(nb_actions, activation="linear")(c)
critic_layer = keras.layers.Dense(1, activation="linear")(c)

model = keras.Model(input_, [actor_layer, critic_layer])
optimizer = Adam(learning_rate=lr)
model.summary()

def LinearSoftmaxPolicy(model, state, nb_actions):
    action_eval, _ = model(state.reshape((1,-1)))
    probs = tf.nn.softmax(action_eval)
    return np.random.choice(nb_actions, 1, p=probs[0].numpy())[0]

def  train(model, experiences):

    gamma = 0.9  # 割引率

    # 現在からエピソード最後までの報酬を計算（後ろから計算）
    if experiences[-1]["done"]:
        # 最後が終わりの場合は全部使える
        G = 0
    else:
        # 最後が終わりじゃない場合は予測値vで補完する
        n_state = np.atleast_2d(experiences[-1]["n_state"])
        _, n_v = model(n_state)
        G = n_v[0][0].numpy()

    # 割引報酬を後ろから計算
    discounted_rewards = []
    for exp in reversed(experiences):
        if exp["done"]:
            G = 0
        G = exp["reward"] + gamma * G
        discounted_rewards.append(G)
    discounted_rewards.reverse()

    # 計算用にnp化して (batch_size,1) の形にする
    discounted_rewards = np.asarray(discounted_rewards).reshape((-1, 1))

    # ベースライン処理
    discounted_rewards -= np.mean(discounted_rewards)  # 報酬の平均を引く

    # データ形式を変形
    state_batch = np.asarray([e["state"] for e in experiences])
    action_batch = np.asarray([e["action"] for e in experiences])

    # アクションをonehotベクトルの形に変形
    onehot_actions = tf.one_hot(action_batch, nb_actions)

    #--- 勾配を計算する
    with tf.GradientTape() as tape:
        action_eval, v = model(state_batch, training=True)

        # π(a|s)を計算
        # 全アクションの確率をだし、選択したアクションの確率だけ取り出す
        # action_probs: [0.2, 0.8] × onehotアクション: [0 ,1] ＝ [0.8] になる
        action_probs = tf.nn.softmax(action_eval)
        selected_action_probs = tf.reduce_sum(onehot_actions * action_probs, axis=1, keepdims=True)

        #--- アドバンテージを計算
        # アドバンテージ方策勾配で使うvは値として使うので、
        # 勾配で計算されないように tf.stop_gradient を使う
        advantage = discounted_rewards - tf.stop_gradient(v)

        # log(π(a|s)) * A(s,a) を計算
        selected_action_probs = tf.clip_by_value(selected_action_probs, 1e-10, 1.0)  # 0にならないようにclip
        policy_loss = tf.math.log(selected_action_probs) * advantage

        #--- Value loss
        # 平均二乗誤差で損失を計算
        value_loss = tf.reduce_mean((discounted_rewards - v) ** 2, axis=1, keepdims=True)

        #--- 方策エントロピー
        entropy = tf.reduce_sum(tf.math.log(selected_action_probs) * selected_action_probs, axis=1, keepdims=True)

        #--- total loss
        value_loss_weight = 0.5
        entropy_weight = 0.1
        loss = -policy_loss + value_loss_weight * value_loss - entropy_weight * entropy

        # 全バッチのlossの平均(ミニバッチ処理?)
        loss = tf.reduce_mean(loss)

    # 勾配を計算し、optimizerでモデルを更新
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss.numpy()

batch_size = 32
experiences = []
losses = []
steps = []
total_rewards = []
success = 0

# 学習ループ
# env = gym.make("CartPole-v0")
for episode in range(300):
    state, _ = env.reset()
    state = np.asarray(state)
    done = False
    truncated = False
    total_reward = 0
    loss = 0.0

    # 1episode
    while not (done or truncated):

        # アクションを決定
        action = LinearSoftmaxPolicy(model, state, nb_actions)

        # 1step進める
        n_state, reward, done, truncated, _ = env.step(action)
        n_state = np.asarray(n_state)
        total_reward += reward

        # 経験を保存する
        experiences.append({
            "state": state,
            "action": action,
            "reward": reward,
            "n_state": n_state,
            "done": done,
        })

        state = n_state

        # batch_size貯まるごとに学習する
        if len(experiences) == batch_size:
            loss += train(model, experiences)
            experiences = []

    losses.append(loss)
    total_rewards.append(total_reward)
    print('episode:{} total_reward:{} loss:{}'.format(episode,total_reward,loss))

plt.plot(total_rewards)
plt.plot(losses)
plt.legend(('reward','losses'))
plt.show()

env = gym.make("CartPole-v1",render_mode="human")
# 5回テストする
for episode in range(5):
    state, _ = env.reset()
    state = np.asarray(state)
    done = False
    truncated = False
    total_reward = 0
    step = 0

    # 1episode
    while not (done or truncated):
        action = LinearSoftmaxPolicy(model, state, nb_actions)
        n_state, reward, done, truncated, _ = env.step(action)
        state = np.asarray(n_state)
        step += 1
        total_reward += reward

    print("{} step, reward: {}".format(step, total_reward))

