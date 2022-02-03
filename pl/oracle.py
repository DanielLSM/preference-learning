import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tf_utils import mlp


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
        output = tf.squeeze(layers.Dense(1)(model))
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
            # print(o0)
            o0 = np.stack([i.tolist() for i in o0])
            o1 = np.stack([i.tolist() for i in o1])

            # print(o0)
            # print(o0, o1, pref_dist, mini_batch_sizes)
            # self.train_mini_batch(np.concatenate(o0, axis=0), np.concatenate(o1, axis=0), pref_dist, mini_batch_sizes)
            self.train_mini_batch(o0, o1, pref_dist, mini_batch_sizes)

            # return loss

    def preference_calculation(self, r1_mean_exp, r2_mean_exp):
        return tf.divide(r1_mean_exp, tf.add(r1_mean_exp, r2_mean_exp))

    @tf.function(experimental_relax_shapes=True)
    def train_mini_batch(self, o0, o1, pref_dist, batch_sizes):
        with tf.GradientTape() as tape:
            # tf.print(o0, output_stream=sys.stdout)
            r1_mean_exp = tf.exp(tf.divide(tf.reduce_sum(self.reward_model(o0, training=True)), batch_sizes[0]))
            r2_mean_exp = tf.exp(tf.divide(tf.reduce_sum(self.reward_model(o1, training=True)), batch_sizes[1]))
            pref_r1 = self.preference_calculation(r1_mean_exp, r2_mean_exp)
            pref_r2 = self.preference_calculation(r2_mean_exp, r1_mean_exp)
            loss = -(pref_dist[0] * tf.math.log(pref_r1) + pref_dist[1] * tf.math.log(pref_r2))
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
