from builtins import enumerate, range
from threading import Thread, Lock
from multiprocessing import Process, Pipe

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
# import scipy.signal
# import time
# import sys

from tf_utils import set_env_seed
from statistics import mean
from oracle import HumanCritic
from ppo import Buffer, PPO, logprobabilities


class Environment(Process):

    def __init__(self, env_idx, child_conn, env, state_size, action_size):
        super(Environment, self).__init__()
        self.env = env
        self.child_conn = child_conn
        self.state_size = state_size
        self.action_size = action_size

    def run(self):
        super(Environment, self).run()
        observation = self.env.reset()
        # state = np.reshape(state, [1, self.state_size])
        self.child_conn.send(observation)
        while True:
            action = self.child_conn.recv()

            observation, reward, done, _ = self.env.step(action)

            # state = np.reshape(state, [1, self.state_size])

            if done:
                observation = self.env.reset()
                # state = np.reshape(state, [1, self.state_size])

            self.child_conn.send([observation, reward, done, _])


# Multiprocessing
num_workers = 8
return_solved_task = 200

# Human Critic
ask_human = True
trajectory_save_frequency = 1
train_reward_model_freq = 1

# Hyperparameters of the PPO algorithm
num_episodes = 1000
max_size_buffer = int(4000 / num_workers)
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
log_means_frequency = 50

# Initialize the environment and get the dimensionality of the
# observation space and the number of possible actions

env_name = "CartPole-v0"
env = gym.make("CartPole-v0")
env = set_env_seed(env, 1)
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n

# Initialize HumanCritic
human_critic = HumanCritic(obs_size=observation_dimensions, action_size=num_actions)

# Initialize the actor and the critic as keras models
ppo_agent = PPO(observation_dimensions, hidden_sizes, num_actions, policy_learning_rate, value_function_learning_rate, clip_ratio, train_policy_iterations, train_value_iterations,
                target_kl)

workers, parent_conns, child_conns, buffers = [], [], [], []
for idx in range(num_workers):
    parent_conn, child_conn = Pipe()
    worker = Environment(idx, child_conn, env, observation_dimensions, num_actions)
    buffer = Buffer(observation_dimensions, max_size_buffer)
    worker.start()
    workers.append(worker)
    buffers.append(buffer)
    parent_conns.append(parent_conn)
    child_conns.append(child_conn)

observations_workers = [0 for _ in range(num_workers)]
observations_new_workers = [0 for _ in range(num_workers)]
rewards_workers = [0 for _ in range(num_workers)]
dones_workers = [0 for _ in range(num_workers)]
episode_return = [0 for _ in range(num_workers)]
episode_length = [0 for _ in range(num_workers)]
trajectories = [[] for _ in range(num_workers)]

for worker_id, parent_conn in enumerate(parent_conns):
    observations_workers[worker_id] = parent_conn.recv()
# print(observations_workers[worker_id])
# import ipdb

# ipdb.set_trace()

episode = 0
current_mean = 0
# Iterate over the number of epochs
while episode < num_episodes:
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    # observation = env.reset()

    if episode % log_means_frequency == 0:
        episode_return_mean, episode_length_mean = [], []

    # trajectory = []

    # Get the logits, action, and take one step in the environment
    # observation = observation.reshape(1, -1)
    observations = np.reshape(observations_workers, [num_workers, observation_dimensions])
    logits, actions = ppo_agent.sample_action(observations)
    # import ipdb
    # ipdb.set_trace()
    # Get the value and log-probability of the action
    value_t = ppo_agent.critic(observations)
    logprobability_t = logprobabilities(logits, actions, num_actions)

    for worker_id, parent_conn in enumerate(parent_conns):
        parent_conn.send(actions[worker_id].numpy())

    for worker_id, parent_conn in enumerate(parent_conns):
        observations_new_workers[worker_id], reward, dones_workers[worker_id], _ = parent_conn.recv()

        episode_return[worker_id] += reward
        episode_length[worker_id] += 1

        # Store obs, act, rew, v_t, logp_pi_t
        # trajectory.append([observation.copy(), observation_new.copy(), action, done])
        # reward = reward if not ask_human else human_critic.reward_model(observation).numpy()
        # import ipdb
        # ipdb.set_trace()
        trajectories[worker_id].append([observations_workers[worker_id].copy(), observations_new_workers[worker_id].copy(), actions[worker_id].numpy(), dones_workers[worker_id]])
        rewards_workers[worker_id] = reward if not ask_human else human_critic.reward_model(observations_workers[worker_id].reshape(1, -1)).numpy()
        # import ipdb
        # ipdb.set_trace()

        buffers[worker_id].store(observations_workers[worker_id], actions[worker_id], rewards_workers[worker_id], value_t[worker_id], logprobability_t[worker_id])
        # import ipdb
        # ipdb.set_trace()
        # print(buffer.observation_buffer)
        if buffers[worker_id].buffer_full():
            # Get values from the buffer
            (
                observation_buffer,
                action_buffer,
                advantage_buffer,
                return_buffer,
                logprobability_buffer,
            ) = buffers[worker_id].get()
            ppo_agent.train_ppo(observation_buffer, action_buffer, advantage_buffer, return_buffer, logprobability_buffer)

        # Update the observation
        observations_workers[worker_id] = observations_new_workers[worker_id]

        # Finish trajectory if reached to a terminal state
        terminal = dones_workers[worker_id]
        if terminal:
            episode_return_mean.append(episode_return[worker_id])
            episode_length_mean.append(episode_length[worker_id])

            last_value = 0 if dones_workers[worker_id] else ppo_agent.critic(observations_workers[worker_id].reshape(1, -1))
            buffers[worker_id].finish_trajectory(last_value)
            if ask_human and episode % trajectory_save_frequency == 0:
                human_critic.add_trajectory(None, None, episode_return[worker_id], trajectories[worker_id])
                human_critic.ask_total_reward()
            if ask_human and episode % train_reward_model_freq == 0:
                human_critic.train_reward_model()

            episode_return[worker_id], episode_length[worker_id], trajectories[worker_id] = 0, 0, []

            if episode % log_means_frequency == 0:
                # Print mean return and length for each epoch
                print(f"\033[1;32m Episode: {episode}. Mean Return: {mean(episode_return_mean)}. Mean Length: {mean(episode_length_mean)}")
                current_mean = mean(episode_return_mean)
            episode += 1
    if current_mean >= return_solved_task:
        print("=============TASK SOLVED=============")
        break

# terminating processes after a while loop
for work in workers:
    work.terminate()
    print('TERMINATED:', work)
    work.join()