# Main file for SPI-CHRL
import warnings
import numpy as np

import lib.grid_mdp as MDP
import lib.plot_data as plot_data
import lib.build as build
import lib.utils as utils



warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)



args = utils.parse_args()

env_type = args.env_type
env = args.env
method = args.method

baseline = args.baseline
d_size = args.d_size
n_wedge = args.n_wedge

N = args.N
j_max = args.j
reps = args.reps
regret = args.regret

for m in [method]:
    for f in [env_type]:
        for e in [env]:
            print(f'----{f}:{e}----\n')

            P = np.load(f'mdp_data/configs/{f}/{e}_P.npy')
            R = np.load(f'mdp_data/configs/{f}/{e}_R.npy')
            init_state = np.load(f'mdp_data/configs/{f}/{e}_init.npy')
            goal_states = np.load(f'mdp_data/configs/{f}/{e}_goal.npy')

            mdp = MDP.mdp(P, R, init_state, goal_states)
            for b in [baseline]:
                try: 
                    pi_b = np.load(f'mdp_data/baselines/{f}_pols/{e}_pib_{b}.npy')
                except:
                    pi_b = build.gen_baseline(mdp, b)

                    try: 
                        if pi_b == None:
                            break
                    except:
                        np.save(f'mdp_data/baselines/{f}_pols/{e}_pib_{b}.npy', pi_b)

                for d in [d_size]:
                    for nw in [n_wedge]:
                        for n in [N]:

                            plot_data.gen_trials(f, e, m, mdp, pi_b, b, d, nw, n, j_max, reps, regret)
                            plot_data.plot_perf(f, e, m, mdp, pi_b, b, d, nw, n, j_max, reps, regret)
                            plot_data.plot_counts_graph(f, e, m, mdp, pi_b, b, d, nw, n, j_max, reps, regret)
                            plot_data.plot_uncertainty_graph(f, e, m, mdp, pi_b, b, d, nw, n, j_max, reps, regret)
                            #plot_data.plot_cvar(fname, ename, m, mdp, pi_b, b, d, nw, n, j_max, reps)




