import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Buffer:
    def __init__(self,
            num_states, num_actions, 
            buffer_capacity=100000, 
    ):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def sample(self, batch_size):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        return (state_batch, action_batch, reward_batch, next_state_batch)

class Ddpg:
    def __init__(self,
        num_states, num_actions,
        upper_bound, lower_bound, 
        batch_size=64,
        std_dev = 0.2,
        critic_lr = 0.002, # Learning rate for actor-critic models
        actor_lr = 0.001,
        gamma = 0.99, # Discount factor for future rewards
        tau = 0.005,  # Used to update target networks
    ):
        # Num of tuples to train on.
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.ou_noise = OUActionNoise(
            mean=np.zeros(1),
            std_deviation=float(std_dev) * np.ones(1))

        # noi = []
        # for _ in range(100):
        #     #x = ou_noise()
        #     x = np.random.normal([1]) * np.sqrt(1e-2)
        #     noi.append(x)
        # 
        # plt.plot(noi)
        # plt.show()
        # exit()

        self.actor_model = self.get_actor(num_states, num_actions)
        self.critic_model = self.get_critic(num_states, num_actions)

        self.target_actor = self.get_actor(num_states, num_actions)
        self.target_critic = self.get_critic(num_states, num_actions)

        # Making the weights equal initially
        self.copyWeights(1.0)
        # self.target_actor.set_weights(self.actor_model.get_weights())
        # self.target_critic.set_weights(self.critic_model.get_weights())

        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        self.last_noise = None


    def get_actor(self, num_states, num_actions):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(num_states,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * self.upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model


    def get_critic(self, num_states, num_actions):
        # State as input
        state_input = layers.Input(shape=(num_states))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(num_actions))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def action(self, state, training=False):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        actions = tf.squeeze(self.actor_model(state))
        actions = actions.numpy()

        if training == True:
            noise = self.ou_noise()
            self.last_noise = noise
            # Adding noise to action
            actions = actions + noise

        # We make sure action is within bounds
        legal_action = np.clip(actions, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)]

    def save(self, filename):
        self.actor_model.save_weights("{}_actor.h5".format(filename))
        self.critic_model.save_weights("{}_critic.h5".format(filename))

    def load(self, filename):
        self.actor_model.load_weights("{}_actor.h5".format(filename))
        self.critic_model.load_weights("{}_critic.h5".format(filename))

    def exists(self, filename):
        return exists("{}_actor.h5".format(filename))


    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.

    def copyWeights(self, tau):
        self.update_target(self.target_actor.variables, self.actor_model.variables, tau)
        self.update_target(self.target_critic.variables, self.critic_model.variables, tau)

    # @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    def update(self, buffer):
        state_batch, action_batch, reward_batch, next_state_batch \
            = buffer.sample(self.batch_size)

        loss = self.learn(state_batch, action_batch, reward_batch, next_state_batch)
        self.copyWeights(self.tau)
        return loss

    @tf.function
    def learn(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value_t = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value_t)
            # actor_loss = -tf.math.reduce_mean(tf.math.square(critic_value_t))

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

        return (actor_loss,critic_loss)


problem = "Pendulum-v1"
env = gym.make(problem)

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

# ou_noise = OUActionNoise(
#     mean=np.zeros(1),
#     std_deviation=float(0.2) * np.ones(1))
# noi = []
# for _ in range(20000):
#     x = ou_noise()
#     #x = np.random.normal([1]) * np.sqrt(1e-2)
#     noi.append(x)
# 
# plt.plot(noi)
# plt.show()
# exit()

# agent = Ddpg(num_states, num_actions, upper_bound, lower_bound, batch_size=2)
# buffer = Buffer(num_states, num_actions,2)
# prev_state = np.array([0.,0.,0.],np.float32)
# action = np.array([0.1],np.float32)
# reward = 0.1
# state = np.array([0.,0.1,0.1],np.float32)
# buffer.record((prev_state, action, reward, state))
# buffer.record((prev_state, action, reward, state))
# losses = [];
# for _ in range(100):
#     actor_loss,critic_value = agent.update(buffer)
#     print(critic_value)
#     losses.append(actor_loss)
# plt.plot(losses)
# plt.show()
# exit()

# layer = tf.keras.layers.Dense(1)
# i = tf.Variable([[0,0.5,0.1],[0,0.5,0.1]])
# print(layer(i))
# exit()

total_episodes = 100

agent = Ddpg(num_states, num_actions, upper_bound, lower_bound, batch_size=64)
buffer = Buffer(num_states, num_actions, 50000)

if not agent.exists('pendulum2'):

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    avg_loss_list = []
    avg_closs_list = []
    noise_list = []

    # Takes about 4 min to train
    for ep in range(total_episodes):
        losses = []
        critic_losses = []

        prev_state = env.reset()
        episodic_reward = 0

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()


            action = agent.action(prev_state, training=True)
            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)

            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward

            actor_loss,critic_loss = agent.update(buffer)
            losses.append(actor_loss)
            critic_losses.append(critic_loss)

            # End this episode when `done` is True
            if done:
                break

            prev_state = state

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        loss = np.mean(losses)
        closs = np.mean(critic_losses)
        print("Episode {:3d} : Avg Reward => {:.2f}  Loss => {:.3e}  Critic => {:.3e}".format(ep+1, avg_reward, loss, closs))
        avg_reward_list.append(avg_reward)
        avg_loss_list.append(loss)
        avg_closs_list.append(closs)
        noise_list.append(agent.last_noise)

    # Plotting graph
    # Episodes versus Avg. Rewards
    avg_loss_list = np.array(avg_loss_list)
    avg_loss_list = (avg_loss_list-np.min(avg_loss_list))/(np.max(avg_loss_list)-np.min(avg_loss_list))* \
        (np.max(avg_reward_list)-np.min(avg_reward_list))+np.min(avg_reward_list)
    avg_closs_list = np.array(avg_closs_list)
    avg_closs_list = (avg_closs_list-np.min(avg_closs_list))/(np.max(avg_closs_list)-np.min(avg_closs_list))* \
        (np.max(avg_reward_list)-np.min(avg_reward_list))+np.min(avg_reward_list)
    plt.plot(avg_reward_list)
    plt.plot(avg_loss_list)
    plt.plot(avg_closs_list)
    plt.legend(['reward','loss','closs'])
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.title("Reward")
    plt.figure()
    plt.plot(noise_list)
    plt.title("Noise")
    plt.show()

    # Save the weights
    agent.save("pendulum2")

    # target_actor.save_weights("pendulum_target_actor.h5")
    # target_critic.save_weights("pendulum_target_critic.h5")
else:
    agent.load("pendulum2")

    # target_actor.load_weights("pendulum_target_actor.h5")
    # target_critic.load_weights("pendulum_target_critic.h5")


eval_steps = 0
for _ in range(5):
    ev_state = env.reset()
    ev_done = False
    while ev_done == False:
        env.render()
        ev_action = agent.action(ev_state)
        ev_state, ev_reward, ev_done, ev_info = env.step(ev_action)
        eval_steps += 1
