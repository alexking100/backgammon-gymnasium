import gymnasium as gym
import time
from itertools import count
import random
import numpy as np
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS, TOKEN

env = gym.make('gym_backgammon:backgammon-v0')
# env = gym.make('gym_backgammon:backgammon-pixel-v0')

# Access the underlying environment to use custom methods
unwrapped_env = env.unwrapped

# Start a new game
agent_color, first_roll, observation = env.reset()

# Take actions in a loop
observation, reward, done, info = env.step(random.choice(list(unwrapped_env.get_valid_actions(first_roll))))
