import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)


def set_env_seed(env, seed):
    env.np_random = np.random.RandomState(seed)
    env.seed(seed)
    return env
