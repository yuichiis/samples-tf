# coding:utf-8
import gym
import numpy as np
import random
import time
from collections import deque
from collections import namedtuple
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

def huber_loss(y_true, y_pred):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.
    clip_value = 1.0
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    #  x = trues - predicts
    #  loss = 0.5 * x^2                  if |x| <= d
    #  loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
    #       = d*|x| - 0.5*d^2
    #       = d*(|x| - 0.5*d)
    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if hasattr(tf, 'select'):
        return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
    else:
        return tf.where(condition, squared_loss, linear_loss)  # condition, true, false

class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        if(len(self.buffer) < batch_size):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        return batch

    def len(self):
        return len(self.buffer)

class TrajectoryBuffer:
    def __init__(self, max_size=1000):
        self._max_size = max_size
        self._buffers = []
        self._last_index = -1
        self._size = 0

    def add(self, transition):
        if len(self._buffers) == 0:
            for field in transition:
                shape = (self._max_size,) + field.shape
                self._buffers.append(np.zeros(shape,dtype=field.dtype))
        else:
            if len(self._buffers) != len(transition):
                raise Exception('unmatch number of fields: Expecting %d but given %d' \
                                (len(self._buffers),len(transition)))
        self._last_index += 1
        if self._last_index >= self._max_size:
            self._last_index = 0
        self._size = max(self._size,self._last_index+1)
        i = 0
        for field in transition:
            self._buffers[i][self._last_index] = field
            i += 1

    def sample(self, batch_size):
        if batch_size > self._size:
            batch_size = self._size
        if batch_size <= 0:
            return []
        if self._size > self._max_size:
            raise Exception('Illegal size %d. max_size is %d'%(self._size,self._max_size))
        indexes = np.random.choice(range(self._size), batch_size,replace=False)
        results = []
        for buffer in self._buffers:
            results.append(buffer[indexes])
        return results

    def len(self):
        return self._size


#a = TrajectoryBuffer(max_size=10)
#for _ in range(20):
#    a.add((np.array(1),))
#print(a.len())
#for _ in range(10000):
#    b = a.sample(9)
#exit()

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done', 'info'))
Trajectory = namedtuple(
    'Trajectory', ('state', 'action', 'next_state', 'reward', 'discount'))

class QNetwork:
    def __init__(self, state_size, action_size, fc_layers):
        self.model = self.build_model(state_size,action_size,fc_layers)
        self.target_model = self.build_model(state_size,action_size,fc_layers)
        self.sync_model()

    def build_model(self, obs_space,action_space,fc_layers):
        layers = [tf.keras.layers.Flatten(input_dim=obs_space)]
        for units in fc_layers:
            layers.append(tf.keras.layers.Dense(units,activation='relu'))
        layers.append(tf.keras.layers.Dense(action_space))
        return tf.keras.models.Sequential(layers)

    def sync_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_q_values(self, state):
        q_values = self.model(tf.expand_dims(tf.constant(state),axis=0))
        q_values = tf.reshape(q_values,[q_values.shape[1]])
        return q_values

    def get_action(self, state):
        q_values = self.get_q_values(state);
        action = tf.argmax(q_values).numpy()
        return action

class DqnAgent:
    def __init__(self,
            network, state_size, action_size,
            gamma=1.0, reward_scale_factor=1.0,learning_rate=1e-3,policy=None,
            target_update_period=1,
            buffer_size=1000):
        self.network = network
        self.policy = policy
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.reward_scale_factor = reward_scale_factor
        self.loss = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.policy = policy
        self.target_update_period = target_update_period
        self.target_update_timer = self.target_update_period
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        #self.replay_buffer = TrajectoryBuffer(max_size=buffer_size)
        self.network.model.compile(loss=self.loss, optimizer=self.optimizer)

    def get_action(self, state):
        action = self.policy.get_action(state)
        return action

    def collect_step(self, *args):
        data = Transition(*args)
        #data = self.format_traject_entry(data)
        self.replay_buffer.add(data)

    def format_traject_entry(self, ts):
        state = ts.state
        action = np.array(ts.action,dtype=np.int32)
        next_state = ts.next_state
        reward = np.array(ts.reward,dtype=np.float32)
        discount = np.array((1 if ts.done == False else 0),dtype=np.float32)
        trajectory = Trajectory(state, action, next_state, reward, discount)
        return trajectory

    def build_gather_indices(self, actions):
        i = 0
        indices = []
        for action in actions:
            indices.append([i,action])
            i += 1
        return tf.constant(indices)

    def build_discounts(self, dones):
        discounts = []
        for done in dones:
            discounts.append(1 if done == False else 0)
        return tf.constant(discounts,dtype=tf.float32)

    def get_trajectory(self, transitions):
        states = [transition.state for transition in transitions]
        actions = [transition.action for transition in transitions]
        rewards = [transition.reward for transition in transitions]
        dones = [transition.done for transition in transitions]
        next_states = [transition.next_state for transition in transitions]

        states = tf.constant(states,dtype=tf.float32)
        rewards = tf.constant(rewards,dtype=tf.float32)
        next_states = tf.constant(next_states,dtype=tf.float32)
        actions = self.build_gather_indices(actions)
        discounts = self.build_discounts(dones)
        return (states, actions, next_states, rewards, discounts)

    def get_trajectory2(self, batch):
        traj = Trajectory(*batch)
        states = tf.constant(traj.state)
        actions = self.build_gather_indices(traj.action)
        next_states = tf.constant(traj.next_state)
        rewards = tf.constant(traj.reward)
        discounts = tf.constant(traj.discount)
        return (states, actions, next_states, rewards, discounts)

    def train(self, batch_size):
        if self.replay_buffer.len() < batch_size:
            return 0
        state_batch = np.zeros((batch_size, 4))
        action_batch = []
        next_state_batch = np.zeros((batch_size, 4))
        next_state_values = np.zeros((batch_size, 1))
        non_final_mask = []

        batch = self.replay_buffer.sample(batch_size)
        states, actions, next_states, rewards, discounts = \
            self.get_trajectory(batch)

        next_q = self.network.target_model.predict(next_states,batch_size=batch_size)
        next_q_values = tf.math.reduce_max(next_q,axis=1)
        q_values = self.reward_scale_factor * rewards + \
                        self.gamma * next_q_values * discounts

        predict_q_values = self.network.model.predict(states,batch_size=batch_size)
        target_values = tf.scatter_nd(actions, q_values, [batch_size,self.action_size])
        q_ones = tf.ones([batch_size])
        ones = tf.ones([batch_size,self.action_size])
        mask = ones - tf.scatter_nd(actions, q_ones, [batch_size,self.action_size])
        target_values += predict_q_values*mask

        history = self.network.model.fit(states,target_values,
            batch_size=batch_size, epochs=1, verbose=0)

        if self.target_update_timer is not None:
            self.target_update_timer -= 1
            if self.target_update_timer <= 0:
                self.network.sync_model()
                self.target_update_timer = self.target_update_period

        return history.history['loss'][0]

class AnnealingEpsGreedyQPolicy:
    def __init__(self, policy, action_size, start=0.9, stop=0.01, decay_rate=0.001):
        self.policy = policy
        self.action_size = action_size
        self.start = start
        self.stop = stop
        self.decay_rate = decay_rate
        self.total_step = 0

    def get_epsilon(self):
        return self.stop + (self.start-self.stop)*np.exp(-self.decay_rate*self.total_step)

    def get_action(self, state):
        epsilon = self.get_epsilon()

        if epsilon <= np.random.uniform(0, 1):
            action = self.policy.get_action(state)
        else:
            action = np.random.choice(range(self.action_size))

        self.total_step += 1
        return action

class BoltzmannQPolicy:
    def __init__(self, tau=1., clip=(-500., 500.)):
        self.tau = tau
        self.clip = clip

    def get_esilon(self,total_step):
        return 0

    def get_action(self, state, total_step, mainQN):   # [C]ｔ＋１での行動を返す
        q_values = mainQN.model.predict(state)[0]
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        return action
#                state, action, next_state, reward, done, info
a = Transition(    0,      0,          0,      0,    0,    0)
print(len(a))

num_episodes = 400
max_number_of_steps = 200
gamma = 1.0#0.99
reward_scale_factor = 1.0
fc_layers = [100]#[16,16]
learning_rate = 1e-3
buffer_size = 10000
batch_size = 32
target_update_period = 200#None#150#100
epsilon_start = 1.0
epsilon_stop = 0.05#0.01
decay_rate = 0.001
env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

q_net = QNetwork(state_size, action_size, fc_layers)
#replay_buffer = ReplayBuffer(max_size=buffer_size)
#policy = GreedyQPolicy()
#policy = BoltzmannQPolicy()
policy = AnnealingEpsGreedyQPolicy(
    q_net, action_size, start=epsilon_start, stop=epsilon_stop, decay_rate=decay_rate)
agent = DqnAgent(
    q_net, state_size=state_size, action_size=action_size,
    gamma=gamma, reward_scale_factor=reward_scale_factor,learning_rate=learning_rate,
    target_update_period=target_update_period,
    policy=policy, buffer_size=buffer_size )

losses_list = []
reward_list = []
episode_steps_list = []
epsilon_list = []
total_step = 0

#for episode in tqdm(range(num_episodes)):
for episode in range(num_episodes):#tqdm(range(num_episodes)):
    start_time = time.perf_counter()
    episode_steps = 0
    episode_reward = 0
    episode_loss = 0
    state = env.reset()
    done = False

    while done != True:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        episode_steps += 1

        episode_reward += reward

        agent.collect_step(state, action, next_state, reward, done, info)
        state = next_state

        episode_loss += agent.train(batch_size)

    if target_update_period is None:
        agent.network.sync_model()

    end_time = time.perf_counter()
    avg_time = (end_time-start_time)/episode_steps*1000
    avg_losses = episode_loss/episode_steps
    losses_list.append(avg_losses)
    reward_list.append(episode_reward)
    episode_steps_list.append(episode_steps)
    epsilon_list.append(policy.get_epsilon())
    q = tf.math.reduce_max(q_net.get_q_values(state))
    print('ep:{} st:{} rw:{} loss:{:.3f} eps:{:.3f} q:{:.2f} {:.0f}ms/st'.format(
        episode+1,episode_steps,episode_reward,avg_losses,policy.get_epsilon(),q,avg_time))

plt.figure()
plt.plot(losses_list)
plt.title('losses')
plt.figure()
plt.plot(episode_steps_list)
plt.title('episode_steps')
plt.figure()
plt.plot(reward_list)
plt.title('reward')
plt.figure()
plt.plot(epsilon_list)
plt.title('epsilon')
plt.show()
