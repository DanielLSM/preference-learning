import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal
import time
import sys


class HumanCritic:
    SIZES = (64, 64)
    BATCH_SIZE = 10
    LEARNING_RATE = 0.00025
    BUFFER_SIZE = 1e5

    def __init__(self, obs_size=3, action_size=2):

        # ===BUFFER===
        self.trajectories = []
        self.preferences = []

        # === MODEL ===
        self.obs_size = obs_size
        self.action_size = action_size
        self.NEURON_SIZE = obs_size**3
        self.init_tf()

    def init_tf(self):
        # ==INPUTS==
        self.input = keras.Input(shape=(self.obs_size, ), dtype=tf.float32)
        # self.input_o1 = keras.Input(shape=(self.obs_size, ), dtype=tf.float32)
        # https://stackoverflow.com/questions/58986126/replacing-placeholder-for-tensorflow-v2
        self.preference = keras.Input(shape=(2), dtype=tf.float32)
        self.batch_sizes = keras.Input(shape=(2), dtype=tf.float32)

        # ==MODELS==
        self.reward_model = self._create_reward_model(self.input)

        # ==OPTIMIZER==
        self.optimizer = keras.optimizers.Adam(learning_rate=self.LEARNING_RATE)

    def _create_reward_model(self, input_obs):
        model = mlp(x=input_obs, sizes=list(self.SIZES))
        output = layers.Dense(1)(model)
        reward_model = keras.Model(inputs=input_obs, outputs=output)
        return reward_model

    def train_reward_model(self):
        if len(self.preferences) < 5:
            return 0
        batch_size = min(len(self.preferences), self.BATCH_SIZE)
        r = np.asarray(range(len(self.preferences)))
        np.random.shuffle(r)
        for i in r[:batch_size]:
            o0, o1, preference = self.preferences[i]
            pref_dist = np.zeros([2], dtype=np.float32)
            if preference < 2:
                pref_dist[preference] = 1.0
            else:
                pref_dist[:] = 0.5
            mini_batch_sizes = [len(o0), len(o1)]
            print(o0)
            # print(o0, o1, pref_dist, mini_batch_sizes)
            # self.train_mini_batch(np.concatenate(o0, axis=0), np.concatenate(o1, axis=0), pref_dist, mini_batch_sizes)
            self.train_mini_batch(o0, o1, pref_dist, mini_batch_sizes)

            # return loss

    def preference_calculation(self, r1_mean_exp, r2_mean_exp):
        return tf.divide(r1_mean_exp, tf.add(r1_mean_exp, r2_mean_exp))

    @tf.function
    def train_mini_batch(self, o0, o1, pref_dist, batch_sizes):
        with tf.GradientTape() as tape:
            tf.print(o0, output_stream=sys.stdout)
            r1_mean_exp = tf.exp(tf.divide(tf.reduce_sum(self.reward_model(o0)), batch_sizes[0]))
            r2_mean_exp = tf.exp(tf.divide(tf.reduce_sum(self.reward_model(o1)), batch_sizes[1]))
            pref_r1 = self.preference_calculation(r1_mean_exp, r2_mean_exp)
            pref_r2 = self.preference_calculation(r2_mean_exp, r1_mean_exp)
            loss = -(pref_dist[0] * tf.log(pref_r1) + pref_dist[1] * tf.log(pref_r2))
        reward_model_grads = tape.gradient(loss, self.reward_model.trainable_variables)
        self.optimizer.apply_gradients(zip(reward_model_grads, self.reward_model.trainable_variables))
        # return loss

    def add_preference(self, o0, o1, preference):
        self.preferences.append([o0, o1, preference])

    def add_trajectory(self, trajectory_env_name, trajectory_seed, total_reward, trajectory):
        self.trajectories.append([trajectory_env_name, trajectory_seed, total_reward, trajectory])

    def ask_total_reward(self):
        if len(self.trajectories) < 2:
            return
        r = np.asarray(range(len(self.trajectories)))
        np.random.shuffle(r)
        t = [self.trajectories[r[0]], self.trajectories[r[1]]]
        if t[0][2] > t[1][2]:
            preference = 0
        elif t[0][2] < t[1][2]:
            preference = 1
        else:
            preference = 2
        os = []
        for i in range(len(t)):
            env_name, seed, total_reward, trijectory = t[i]
            o = []

            for j in range(len(trijectory)):
                o.append(trijectory[j][1])

            os.append(o)

        self.add_preference(os[0], os[1], preference)


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros((size, observation_dimensions), dtype=np.float32)
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0
        self.storage_count = 0
        self.size = size

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1
        self.storage_count += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(deltas, self.gamma * self.lam)
        self.return_buffer[path_slice] = discounted_cumulative_sums(rewards, self.gamma)[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        self.storage_count = 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )

    def buffer_full(self):
        return True if self.storage_count == self.size else False


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(tf.one_hot(a, num_actions) * logprobabilities_all, axis=1)
    return logprobability


# Sample action from actor
@tf.function
def sample_action(observation):
    logits = actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action


# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer):

    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(logprobabilities(actor(observation_buffer), action_buffer) - logprobability_buffer)
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_buffer, min_advantage))
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(logprobability_buffer - logprobabilities(actor(observation_buffer), action_buffer))
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer))**2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))


# Human Critic
trajectory_save_frequency = 1
train_reward_model_freq = 1

# Hyperparameters of the PPO algorithm
num_episodes = 5000
max_size_buffer = 4000
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01
hidden_sizes = (64, 64)

# True if you want to render the environment
render = False
render_frequency_episode = 100
log_means_frequency = 50

# Initialize the environment and get the dimensionality of the
# observation space and the number of possible actions
env = gym.make("CartPole-v0")
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n

# Initialize the buffer
buffer = Buffer(observation_dimensions, max_size_buffer)
human_critic = HumanCritic(obs_size=observation_dimensions, action_size=num_actions)

# Initialize the actor and the critic as keras models
observation_input = keras.Input(shape=(observation_dimensions, ), dtype=tf.float32)
logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
actor = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
critic = keras.Model(inputs=observation_input, outputs=value)

# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)


def train_ppo():
    # Get values from the buffer
    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = buffer.get()

    # Update the policy and implement early stopping using KL divergence
    for _ in range(train_policy_iterations):
        kl = train_policy(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer)
        if kl > 1.5 * target_kl:
            # Early Stopping
            break

    # Update the value function
    for _ in range(train_value_iterations):
        train_value_function(observation_buffer, return_buffer)


# Initialize the observation, episode return and episode length
observation, episode_return, episode_length = env.reset(), 0, 0

# Iterate over the number of epochs
for episode in range(num_episodes):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    observation = env.reset()
    episode_return, episode_length = 0, 0
    if episode % log_means_frequency == 0:
        episode_return_mean, episode_length_mean = [], []
    done = False

    trajectory = []

    # Iterate over the steps of each epoch
    while not done:
        if render and episode % render_frequency_episode == 0:
            env.render()

        # Get the logits, action, and take one step in the environment
        observation = observation.reshape(1, -1)
        logits, action = sample_action(observation)
        observation_new, reward, done, _ = env.step(action[0].numpy())
        episode_return += reward
        episode_length += 1

        # Get the value and log-probability of the action
        value_t = critic(observation)
        logprobability_t = logprobabilities(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        trajectory.append([observation.copy(), observation_new.copy(), action, done])
        buffer.store(observation, action, reward, value_t, logprobability_t)
        # print(buffer.observation_buffer)
        if buffer.buffer_full():
            train_ppo()

        # Update the observation
        observation = observation_new

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal:
            episode_return_mean.append(episode_return)
            episode_length_mean.append(episode_length)
            last_value = 0 if done else critic(observation.reshape(1, -1))
            buffer.finish_trajectory(last_value)
            if episode % trajectory_save_frequency == 0:
                human_critic.add_trajectory(None, None, episode_return, trajectory)
                human_critic.ask_total_reward()
            if episode % train_reward_model_freq == 0:
                human_critic.train_reward_model()

    if episode % log_means_frequency == 0:
        # Print mean return and length for each epoch
        print(
            f" Episode: {episode + 1}. Mean Return: {sum(episode_return_mean)/max(len(episode_return_mean),1)}. Mean Length: {sum(episode_length_mean)/max(len(episode_length_mean),1)}"
        )
