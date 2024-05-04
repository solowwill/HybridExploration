# Main file for SPI-CHRL

import gym
import warnings
import numpy as np

import lib.grid_mdp as GRIDMDP
import lib.build as build
import lib.utils as utils

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)

NUM_ACTIONS = 4
fname='25x25_lava'


obstacle_map = build.obstacle_map(fname)

init_state = (13,0)

mdp = GRIDMDP.grid_mdp(obstacle_map, NUM_ACTIONS, init_state)

np.save(f'mdp_data/configs/grid_lava/{fname}_P.npy', mdp.P)
np.save(f'mdp_data/configs/grid_lava/{fname}_R.npy', mdp.R)
np.save(f'mdp_data/configs/grid_lava/{fname}_init.npy', mdp.init_state)
np.save(f'mdp_data/configs/grid_lava/{fname}_goal.npy', mdp.goal_states)

'''env = gym.make('SimpleGrid-v0', 
        obstacle_map=obstacle_map)

env.reset()
env.render()
'''

