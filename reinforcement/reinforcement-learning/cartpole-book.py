import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from collections import deque

NUM_EPISODES = 500
MAX_STEPS = 200
GAMMA = 0.99
WARMUP = 10
E_START = 1.0
E_STOP = 0.01
E_DECAY_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 32

class QNetwork:
  def __init__(self, state_size, action_size):
    self.model = Sequential()
    self.model.add(Dense(16, activation='relu', input_dim=state_size))
    self.model.add(Dense(16, activation='relu'))
    self.model.add(Dense(16, activation='relu'))
    self.model.add(Dense(action_size, activation='linear'))
    self.model.compile(loss=Huber(), optimizer=Adam(learning_rate=0.001))

class Memory:
  def __init__(self, memory_size):
    self.buffer = deque(maxlen=memory_size)

  def add(self, experience):
    self.buffer.append(experience)

  def sample(self, batch_size):
    idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
    return [self.buffer[i] for i in idx]

  def __len__(self):
    return len(self.buffer)

env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

main_qn = QNetwork(state_size, action_size)
target_qn = QNetwork(state_size, action_size)
memory = Memory(MEMORY_SIZE)

state = env.reset()
print(state.dtype)
state = np.reshape(state, [1,state_size])

sampling_steps = 100
n_steps = 0
total_step = 0
success_count = 0
for episode in range(1, NUM_EPISODES+1):
  step = 0
  target_qn.model.set_weights(main_qn.model.get_weights())

  for _ in range(1, MAX_STEPS+1):
    step += 1
    total_step += 1
    epsilon = E_STOP + (E_START-E_STOP)*np.exp(-E_DECAY_RATE*total_step)

    if epsilon > np.random.rand():
      action = env.action_space.sample()
    else:
      action = np.argmax(main_qn.model.predict(state)[0])

    next_state, _, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1,state_size])
    if done:

      if step >= 190:
        success_count += 1
        reward = 1
      else:
        success_count = 0
        reward = 0

      next_state = None#np.zeros(state.shape)
      if step > WARMUP:
        memory.add((state, action, reward, next_state))

    else:
      reward = 0
      if step > WARMUP:
        memory.add((state, action, reward, next_state))

      state = next_state

    if len(memory) >= BATCH_SIZE:
#      inputs  = np.zeros((BATCH_SIZE, 4))
#      targets = np.zeros((BATCH_SIZE, 2))
#
#      minibatch = memory.sample(BATCH_SIZE)
#      for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
#        inputs[i] = state_b
#        if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
#          target = reward_b + GAMMA*np.amax(target_qn.model.predict(next_state_b)[0])
#        else:
#          target = reward_b
#
#        #print('start predict')
#        targets[i] = main_qn.model.predict(state_b)
#        #print('end predict')
#        targets[i][action_b] = target
#
#      #print('start fit')
#      main_qn.model.fit(inputs, targets, batch_size=BATCH_SIZE, epochs=1, verbose=0)
#      #print('end fit')

##########################################################
      inputs = np.zeros((BATCH_SIZE, 4))
      action_batch = []
      next_state_batch = np.zeros((BATCH_SIZE, 4))
      next_state_values = np.zeros((BATCH_SIZE, 1))
      non_final_mask = []

      mini_batch = memory.sample(BATCH_SIZE)

      n_non_final = 0
      for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            if next_state_b is not None:
                n_non_final += 1
                non_final_mask.append(True)
                next_state_batch[i:i + 1] = next_state_b
            else:
                non_final_mask.append(False)
            next_state_values[i] = reward_b
            action_batch.append(action_b)


        # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
        #retmainQs = self.model.predict(next_state_b)[0]
        #next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
        #target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]
      #next_state_values[non_final_mask] = next_state_values[non_final_mask] + \
      #         np.expand_dims(GAMMA*np.max(
      #          target_qn.model.predict(next_state_batch[non_final_mask]) ,axis=1),-1)

      #targets = main_qn.model.predict(inputs)    # Qネットワークの出力
      #targets[:,action_batch] = next_state_values
      next_state_values[non_final_mask] +=  \
            np.expand_dims(GAMMA*np.max(
                target_qn.model.predict(next_state_batch[non_final_mask],
                    batch_size=BATCH_SIZE) ,axis=1),-1)

      targets = main_qn.model.predict(inputs, batch_size=BATCH_SIZE)    # Qネットワークの出力
        #targets[:,action_batch] = next_state_values
      for target,action,value in zip(targets,action_batch,next_state_values):
        target[action] = value

      #n_steps += 1
      #if(n_steps >= sampling_steps):
        #n_steps = 0
        #print('n_non_final:',n_non_final)
        #print('targets:')
        #print(targets)
        #print('targets: min=',np.min(targets),'max=',np.max(targets))


      main_qn.model.fit(inputs, targets, batch_size=BATCH_SIZE,epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定

###########################################################

    if done:
      break

  print('episode: {}, steps: {}, epsilon: {:.4f}'.format(episode, step, epsilon))

  if success_count >= 5:
    break

  state = env.reset()
  state = np.reshape(state, [1, state_size])
