# Main file for SPI-CHRL
print('Starting imports...')
import warnings
import numpy as np
import matplotlib.pyplot as plt 
print('First few imports')

import grid_mdp as MDP
print('second import')
import build as build
import spi_hrl as spi_hrl
import active_exploration as ex
import spibb as SPIBB
import plot_data as plot_data
print('last import')

import argparse

print('got thru imorts')

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--env_type", type=str, default="highway", help="Highway/AgAid")
parser.add_argument("--env", type=str, default="highway", help="intervention[num]_action[num]")
parser.add_argument("--baseline", type=float, default=.8, help="Baseline Performance")


args = parser.parse_args()

print('got thru args')

baseline_perf = args.baseline
d_size = 100
n_wedge = 10
N = 10
j_max = 50
reps = 5
fname = args.env_type
ename = args.env


# Declare the environment types
env_type = ['highway', 'agaid', 'grid', 'grid_lava']
envs = [['highway'],
        ['intvn28_act4_prec1'],
        ['5x5','7x8'],
        ['5x5_lava']]

# possible methods of exploration: pi_b, perturb, spibb
method = ['val_ex']
#method = ['pi_ex_plan']

baselines = [.25]
d_sizes = [5]
n_wedges = [5]
Ns = [1]

# Go through all the envs and env types
for m in method:
    for i in range(len(env_type)):
        fname = env_type[i]
        if env_type[i] == 'agaid' or env_type[i] == 'highway' or env_type[i] == 'grid':
            continue

        for j in range(len(envs[i])):
            ename=envs[i][j]

            print(f'----{fname}:{ename}----\n')
            # Load and create the MDP
            P = np.load(f'mdp_data/configs/{fname}/{ename}_P.npy')
            R = np.load(f'mdp_data/configs/{fname}/{ename}_R.npy')
            init_state = np.load(f'mdp_data/configs/{fname}/{ename}_init.npy')
            goal_states = np.load(f'mdp_data/configs/{fname}/{ename}_goal.npy')

            mdp = MDP.mdp(P, R, init_state, goal_states)
            for b in baselines:
                try: 
                    pi_b = np.load(f'mdp_data/baselines/{fname}_pols/{ename}_pib_{b}.npy')
                except:
                    print('Generating baseline...')
                    pi_b = build.gen_baseline(mdp, b)

                    try: 
                        if pi_b == None:
                            print('No Valid pi b found')
                            break
                    except:
                        np.save(f'mdp_data/baselines/{fname}_pols/{ename}_pib_{b}.npy', pi_b)

                for d in d_sizes:
                    for nw in n_wedges:
                        for n in Ns:

                            plot_data.gen_trials(fname, ename, m, mdp, pi_b, b, d, nw, n, j_max, reps)
                            plot_data.plot_perf(fname, ename, m, mdp, pi_b, b, d, nw, n, j_max, reps)
                            plot_data.plot_counts_graph(fname, ename, m, mdp, pi_b, b, d, nw, n, j_max, reps)
                            plot_data.plot_uncertainty_graph(fname, ename, m, mdp, pi_b, b, d, nw, n, j_max, reps)
                            #plot_data.plot_cvar(fname, ename, m, mdp, pi_b, b, d, nw, n, j_max, reps)




