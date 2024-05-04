# Main file for SPI-CHRL

import gymnasium
import warnings
import numpy as np

import lib.build as build


warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)

NUM_ACTIONS = 4

fname='merge'
env = gymnasium.make(
    f'{fname}-v0'
)

mdp = env.to_finite_mdp()

# Create the transitions
P = np.zeros((mdp.transition.shape[0],mdp.transition.shape[1], mdp.transition.shape[0]))
for i in range(mdp.transition.shape[0]):
    for j in range(mdp.transition.shape[1]):
        P[i,j,mdp.transition[i,j]] = 1

# Normalize the reward
R = build.min_max_norm(mdp.reward)

# Set the init state
init_state = mdp.state

# set the goal states
goal_states = np.arange(mdp.transition.shape[0])[mdp.terminal]


np.save(f'mdp_data/highway/{fname}_P',P)
np.save(f'mdp_data/highway/{fname}_R',R)
np.save(f'mdp_data/highway/{fname}_init', init_state)
np.save(f'mdp_data/highway/{fname}_goal',goal_states)