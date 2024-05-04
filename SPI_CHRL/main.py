# Main file for SPI-CHRL
import sys

import gymnasium as gym
import gym_simplegrid
import numpy as np
import mdp as MDP
import regret
import matplotlib.pyplot as plt 
import build
import spi_chrl as SPI_CHRL
import warnings
warnings.filterwarnings("ignore")

NUM_ACTIONS = 4

obstacle_map = [
        "000",
        "000",
        "000"
]

'''obstacle_map = [
        "00000001",
        "20012100",
        "00000010",
        "01002000",
        "00000000",
        "20011001",
        "00200000",
        "01001003"
    ]'''

# SET THE INITIAL SEED
np.random.seed(1)

init_state = (0,0)

mdp = MDP.mdp(obstacle_map, NUM_ACTIONS, init_state)

np.set_printoptions(precision=3)

policy, val = MDP.value_improvement(mdp) 

print(policy)
print

policy, q = MDP.q_improvement(mdp)

sys.exit(0)

j = MDP.iter_q_constraint_eval(mdp, policy)
v = MDP.iter_v_reward_eval(mdp,policy)

#print(policy)
D = build.build_D(mdp, policy)
print(D)
C = np.array([])
N = 1
N_wedge = 5
epsilon=.01

rand_pol = np.ones((mdp.S,mdp.A)) / mdp.A

spi_chrl = SPI_CHRL.spi_chrl(mdp, rand_pol, D, C, N, N_wedge, epsilon)

# Running SPI_CHRL
spi_chrl.run()





env = gym.make(
    'SimpleGrid-v0', 
    obstacle_map=obstacle_map,
    start = init_state, 
    render_mode='human'
)


obs, info = env.reset()
done = env.unwrapped.done

for _ in range(50):
    if done:
        break
    action = np.random.choice(mdp.action_space, 1, p=policy[env.to_s(obs)])[0]
    obs, reward, done, _, info = env.step(action)
env.close()

