# This file computes everything we need for constraint inference
import numpy as np
import cvxpy as cp
import mdp as MDP


def build_D(mdp, policy):

    trajectories = np.ndarray((0,5))

    for _ in range(5):
        sarcs = mdp.sample_trajectory(policy)
        sarcs = sarcs[ sarcs[:,MDP.C] <= 0]


        trajectories = np.concatenate((trajectories, sarcs),axis=0)

    return trajectories
