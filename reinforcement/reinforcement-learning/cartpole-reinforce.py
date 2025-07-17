import numpy as np
from collections import deque
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import gym
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")

# 環境の状態の形式(shape)
obs_shape = env.observation_space.shape

# 環境の取りうるアクション数
nb_actions = env.action_space.n

lr = 0.01  # 学習率
gamma = 0.9  # 割引率

c = input_ = keras.layers.Input(shape=obs_shape)
c = keras.layers.Dense(10, activation="relu")(c)
c = keras.layers.Dense(10, activation="relu")(c)
c = keras.layers.Dense(nb_actions, activation="softmax")(c)
model = keras.Model(input_, c)
model.compile(optimizer=Adam(lr=lr))
model.summary()

def ProdAction(model, state, nb_actions):
    action_eval = model(state.reshape((1,-1)))
    probs = tf.nn.softmax(action_eval)
    return np.argmax(probs[0].numpy())

def LinearSoftmaxPolicy(model, state, nb_actions):
    action_eval = model(state.reshape((1,-1)))
    probs = tf.nn.softmax(action_eval)
    return np.random.choice(nb_actions, 1, p=probs[0].numpy())[0]

def SoftmaxPolicy(model, state, nb_actions):
    action_probs = model(state.reshape((1,-1)))[0].numpy()
    return np.random.choice(nb_actions, 1, p=action_probs)[0]

def  train(model, experiences):
    # データ形式を変形
    state_batch = np.asarray([e["state"] for e in experiences])
    action_batch = np.asarray([e["action"] for e in experiences])
    reward_batch = np.asarray([e["G"] for e in experiences])
    # print(reward_batch)

    # アクションは one_hot ベクトルにする
    one_hot_actions = tf.one_hot(action_batch, nb_actions)

    # (a)報酬は正規化する
    # (正規化しないと学習が安定しませんでした)
    # (softmax層と相性が悪いから？)
    # reward_batch = StandardScaler().fit_transform(reward_batch.reshape((-1, 1))).flatten()
    reward_batch -= np.mean(reward_batch)  # 報酬の平均を引く
    print(reward_batch)

    # 勾配を計算する
    with tf.GradientTape() as tape:
        # 現在の戦略を取得
        action_probs = model(state_batch, training=True)  # Forward pass

        # (1) 選択されたアクションの確率を取得
        selected_action_probs = tf.reduce_sum(one_hot_actions * action_probs, axis=1)

        # log(0) 回避用
        clipped = tf.clip_by_value(selected_action_probs, 1e-10, 1.0)

        # (2) 期待値「log( π(a|s) ) × Q」を計算
        loss = tf.math.log(clipped) * reward_batch

        # (3) experiences すべての期待値の合計が最大となるように損失を設定
        # 最大値がほしいので負の値にしています
        # 追記：ミニバッチ処理の場所になるので合計ではなく平均のほうが正しい
        loss = -tf.reduce_sum(loss)

    # 勾配を元にoptimizerでモデルを更新
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss.numpy()  # 確認用

losses = []
steps = []
success = 0

# 学習ループ
for episode in range(300):
    state = np.asarray(env.reset())
    done = False
    total_reward = 0
    experiences = []

    # 1episode
    while not done:

        # アクションを決定
        action = SoftmaxPolicy(model, state, nb_actions)

        # 1step進める
        n_state, reward, done, _ = env.step(action)
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

    # 現在からエピソード最後までの報酬を計算
    for i,exp in enumerate(experiences):
        G = 0
        t = 0
        for j in range(i, len(experiences)):
            G += (gamma ** t) * experiences[j]["reward"]
            t += 1
        exp["G"] = G

    # 1エピソード毎に学習（中身は後述）
    loss = train(model, experiences)
    steps.append(len(experiences))
    losses.append(loss)
    print('episode:{} step:{} total_reward:{} loss:{}'.format(episode,len(experiences),total_reward,loss))
    if total_reward>=200:
        success += 1
        if success>=30:
            break
    else:
        success = 0

plt.plot(steps)
plt.plot(losses)
plt.legend(('steps','losses'))
plt.show()

# 5回テストする
for episode in range(5):
    state = np.asarray(env.reset())
    env.render()
    done = False
    total_reward = 0

    step = 0
    # 1episode
    while not done:
        action = ProdAction(model, state, nb_actions)
        n_state, reward, done, _ = env.step(action)
        state = np.asarray(n_state)
        env.render()
        step += 1
        total_reward += reward

    print("{} step, reward: {}".format(step, total_reward))