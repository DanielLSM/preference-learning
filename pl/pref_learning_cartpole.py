import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
# import scipy.signal
# import time
# import sys

from oracle import HumanCritic
from ppo import Buffer, PPO, logprobabilities

# Human Critic
ask_human = True
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
ppo_agent = PPO(observation_dimensions, hidden_sizes, num_actions, policy_learning_rate, value_function_learning_rate, clip_ratio, train_policy_iterations, train_value_iterations,
                target_kl)

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
        import ipdb
        ipdb.set_trace()
        reward = reward if not ask_human else human_critic.reward_model([observation]).numpy()
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
            ppo_agent.train_ppo(observation_buffer, action_buffer, advantage_buffer, return_buffer, logprobability_buffer)

        # Update the observation
        observation = observation_new

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal:
            episode_return_mean.append(episode_return)
            episode_length_mean.append(episode_length)
            last_value = 0 if done else ppo_agent.critic(observation.reshape(1, -1))
            buffer.finish_trajectory(last_value)
            if ask_human and episode % trajectory_save_frequency == 0:
                human_critic.add_trajectory(None, None, episode_return, trajectory)
                human_critic.ask_total_reward()
            if ask_human and episode % train_reward_model_freq == 0:
                human_critic.train_reward_model()

    if episode % log_means_frequency == 0:
        # Print mean return and length for each epoch
        print(
            f" Episode: {episode + 1}. Mean Return: {sum(episode_return_mean)/max(len(episode_return_mean),1)}. Mean Length: {sum(episode_length_mean)/max(len(episode_length_mean),1)}"
        )
