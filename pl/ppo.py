import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.signal

from tf_utils import mlp


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


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def logprobabilities(logits, a, num_actions):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the self.actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(tf.one_hot(a, num_actions) * logprobabilities_all, axis=1)
    return logprobability


class PPO:

    def __init__(self, observation_dimensions, hidden_sizes, num_actions, policy_learning_rate,
                 value_function_learning_rate, clip_ratio, train_policy_iterations, train_value_iterations,
                 target_kl) -> None:

        # Initialize the self.actor and the self.critic as keras models
        observation_input = keras.Input(shape=(observation_dimensions, ), dtype=tf.float32)
        logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
        self.actor = keras.Model(inputs=observation_input, outputs=logits)
        value = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
        self.critic = keras.Model(inputs=observation_input, outputs=value)

        self.observation_dimensions = observation_dimensions
        self.num_actions = num_actions
        self.clip_ratio = clip_ratio
        self.train_policy_iterations = train_policy_iterations
        self.train_value_iterations = train_value_iterations
        self.target_kl = target_kl

        # Initialize the policy and the value function optimizers
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
        self.value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

    # Sample action from self.actor
    @tf.function
    def sample_action(self, observation):
        logits = self.actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action

    # Train the policy by maxizing the PPO-Clip objective
    @tf.function
    def train_policy(self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                logprobabilities(self.actor(observation_buffer), action_buffer, self.num_actions) -
                logprobability_buffer)
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_buffer, min_advantage))
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        kl = tf.reduce_mean(logprobability_buffer -
                            logprobabilities(self.actor(observation_buffer), action_buffer, self.num_actions))
        kl = tf.reduce_sum(kl)
        return kl

    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((return_buffer - self.critic(observation_buffer))**2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))

    def train_ppo(
        self,
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ):

        # Update the policy and implement early stopping using KL divergence
        for _ in range(self.train_policy_iterations):
            kl = self.train_policy(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer)
            if kl > 1.5 * self.target_kl:
                # Early Stopping
                break

        # Update the value function
        for _ in range(self.train_value_iterations):
            self.train_value_function(observation_buffer, return_buffer)


if __name__ == '__main__':

    import gym
    import time
    import sys

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

    buffer = Buffer(observation_dimensions, max_size_buffer)

    ppo_agent = PPO(observation_dimensions, hidden_sizes, num_actions, policy_learning_rate,
                    value_function_learning_rate, clip_ratio)

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
            logits, action = ppo_agent.sample_action(observation)
            observation_new, reward, done, _ = env.step(action[0].numpy())
            episode_return += reward
            episode_length += 1

            # Get the value and log-probability of the action
            value_t = ppo_agent.critic(observation)
            logprobability_t = logprobabilities(logits, action, num_actions)

            # Store obs, act, rew, v_t, logp_pi_t
            trajectory.append([observation.copy(), observation_new.copy(), action, done])
            buffer.store(observation, action, reward, value_t, logprobability_t)
            # print(buffer.observation_buffer)
            if buffer.buffer_full():
                # Get values from the buffer
                (
                    observation_buffer,
                    action_buffer,
                    advantage_buffer,
                    return_buffer,
                    logprobability_buffer,
                ) = buffer.get()
                ppo_agent.train_ppo(observation_buffer, action_buffer, advantage_buffer, return_buffer,
                                    logprobability_buffer)

            # Update the observation
            observation = observation_new

            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal:
                episode_return_mean.append(episode_return)
                episode_length_mean.append(episode_length)
                last_value = 0 if done else ppo_agent.critic(observation.reshape(1, -1))
                buffer.finish_trajectory(last_value)

        if episode % log_means_frequency == 0:
            # Print mean return and length for each epoch
            print(
                f" Episode: {episode + 1}. Mean Return: {sum(episode_return_mean)/max(len(episode_return_mean),1)}. Mean Length: {sum(episode_length_mean)/max(len(episode_length_mean),1)}"
            )
