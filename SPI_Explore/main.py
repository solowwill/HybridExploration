import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import sys
import copy

# Utility functions (arg parse)
class utils:
    @staticmethod
    def parse_args():
        
        parser = argparse.ArgumentParser()

        # Envs 
        # ['highway', 'agaid', 'grid', 'grid_lava']
        parser.add_argument("--env_type", type=str, default="grid_lava", help="grid_lava/grid/highway/agaid")

        # Env Types 
        #    [['highway', 'highway-fast', 'merge', 'roundabout'],
        #    ['intvn28_act4_prec1'],
        #    ['5x5','7x8'],
        #    ['5x5_lava', '7x8_lava]]
        parser.add_argument("--env", type=str, default="5x5_lava")

        # SPIBB Parameters
        parser.add_argument("--baseline", type=float, default=.5, help="Baseline Performance")
        parser.add_argument("--d_size", type=int, default=10, help="Init Dataset Size")
        parser.add_argument("--n_wedge", type=int, default=5, help="N Wedge Uncertainty Parameter")
        
        # SPI-HRL Parameters
        parser.add_argument("--method", type=str, default="uncertainty_v2", help="Exploration Method")
        parser.add_argument("--N", type=int, default=1, help="Number of online episodes")
        parser.add_argument("--reps", type=int, default=5, help="Number of Trials")
        parser.add_argument("--regret", type=float, default=.5, help="Regret Exploration Budget")
        parser.add_argument("--j", type=float, default=50, help="Max Episodes")

        # Change initial-final state
        parser.add_argument("--grid", type=bool, default=False, help="If grid, shuffle init/final states")

        # Reward parameters
        parser.add_argument("--movement_reward", type=float, default=0)
        parser.add_argument("--goal_reward", type=float, default=1)
        parser.add_argument("--obstacle_reward", type=float, default=-1)

        # Path to data
        parser.add_argument("--path", type=str, default="")

        args = parser.parse_args()
        return args


    # Perform min max normalization on an array
    @staticmethod
    def min_max_norm(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


    # Perform Q-Evaluation given an MDP and a policy
    @staticmethod
    def q_eval(mdp, policy, epsilon=.01):

        q = np.zeros((mdp.S, mdp.A))
        
        while True:

            old_q = q

            # Perform Bellman Backups depending on if the reward is R(s,a,s') or R(s,a)
            if len(mdp.R.shape) == 3:
                q = np.sum(mdp.P*mdp.R,axis=-1) + mdp.gamma * np.sum(mdp.P * np.sum(policy * q, axis=-1), axis=-1)
                q[mdp.goal_states,:] = 0
            else:
                q = mdp.R + mdp.gamma * np.sum(mdp.P * np.sum(policy * q, axis=-1), axis=-1)
                q[mdp.goal_states,:] = 0

            if np.max(np.abs(old_q-q)) < epsilon:
                return q


    # Perform Q-Evaluation given an MLE MDP and a policy
    @staticmethod
    def approx_q_eval(mdp, P_hat, R_hat, policy, epsilon=.01):

        q = np.zeros((P_hat.shape[0],P_hat.shape[1]))
        
        for _ in range(1000):

            old_q = q

            # Perform Bellman Backups depending on if the reward is R(s,a,s') or R(s,a)
            # On the MLE MDP
            if len(R_hat.shape) == 3:
                q = np.sum(P_hat*R_hat,axis=-1) + mdp.gamma * np.sum(P_hat * np.sum(policy * q, axis=-1), axis=-1)
                q[mdp.goal_states,:] = 0
            else:
                q = R_hat + mdp.gamma * np.sum(P_hat * np.sum(policy * q, axis=-1), axis=-1)
                q[mdp.goal_states,:] = 0

            if np.max(np.abs(old_q-q)) < epsilon:
                return q


    # Perform Value Improvement given an MDP
    @staticmethod 
    def value_improvement(mdp, epsilon=.01):

        v = np.zeros(mdp.S)
        
        while True:

            old_v = v

            # Perform Bellman Backups depending on if the reward is R(s,a,s') or R(s,a)
            if len(mdp.R.shape) == 3:
                v = np.max(np.sum( mdp.P * (mdp.R + mdp.gamma * v), axis=-1), axis=-1)
                v[mdp.goal_states] = 0
            else:
                v = np.max(np.sum( mdp.P * (mdp.R[:,:,np.newaxis] + mdp.gamma * v), axis=-1), axis=-1)
                v[mdp.goal_states] = 0

            if np.max(np.abs(old_v-v)) < epsilon:
                break

        # Determine the best actions with bellman backups 
        if len(mdp.R.shape) == 3:
            action_reward = np.sum(mdp.P * (mdp.R + mdp.gamma * v), axis=-1)
        else:
            action_reward = np.sum(mdp.P * (mdp.R[:,:,np.newaxis] + mdp.gamma * v), axis=-1)

        # Compute the optimal policy
        best_actions = (action_reward - np.max(action_reward, axis=1)[:,np.newaxis]) == 0
        policy = best_actions * (1 / np.sum( best_actions, axis=1)[:,np.newaxis])

        return policy, v


    # Perform Value Iteration given an MDP and a policy
    @staticmethod
    def v_eval(mdp, policy, epsilon=.01):

        v = np.zeros(mdp.S)
        
        while True:
            
            old_v = v

            # Perform Bellman Backups depending on if the reward is R(s,a,s') or R(s,a)
            if len(mdp.R.shape) == 3:
                v = np.sum( np.sum( mdp.P * (mdp.R + mdp.gamma * v), axis=-1) * policy, axis=-1)
                v[mdp.goal_states] = 0
            else:
                v = np.sum( np.sum( mdp.P * (mdp.R[:,:,np.newaxis] + mdp.gamma * v), axis=-1) * policy, axis=-1)
                v[mdp.goal_states] = 0

            if np.max(np.abs(old_v-v)) < epsilon:
                return v


    # Perform Q Improvement given an MDP
    @staticmethod 
    def q_improvement(mdp, epsilon=.01):

        q = np.zeros((mdp.S,mdp.A))
        
        while True:

            old_q = q

            # Perform Bellman Backups depending on if the reward is R(s,a,s') or R(s,a)
            if len(mdp.R.shape) == 3:
                q = np.sum(mdp.P * (mdp.R + mdp.gamma * np.max(q, axis=-1)), axis=-1)
                q[mdp.goal_states,:] = 0
            else:
                q = np.sum(mdp.P * (mdp.R[:,:,np.newaxis] + mdp.gamma * np.max(q, axis=-1)), axis=-1)
                q[mdp.goal_states,:] = 0

            if np.max(np.abs(old_q-q)) < epsilon:
                break
                
        # Determine the best actions with bellman backups 
        if len(mdp.R.shape) == 3:
            action_reward = np.sum(mdp.P * (mdp.R + mdp.gamma * np.max(q, axis=-1)), axis=-1)
        else:
            action_reward = np.sum(mdp.P * (mdp.R[:,:,np.newaxis] + mdp.gamma * np.max(q, axis=-1)), axis=-1)

        # Compute the optimal policy
        best_actions = (action_reward - np.max(action_reward, axis=1)[:,np.newaxis]) == 0
        policy = best_actions * (1 / np.sum( best_actions, axis=1)[:,np.newaxis])

        return policy, q


    # Approximate Q Improvement given an MLE MDP
    @staticmethod 
    def approx_q_improvement(mdp, P_hat, R_hat, epsilon=.01):

        q = np.zeros((P_hat.shape[0],P_hat.shape[1]))
        
        for _ in range(1000):
            # Store the current value function
            old_q = q

            # Perform Bellman Backups depending on if the reward is R(s,a,s') or R(s,a)
            if len(R_hat.shape) == 3:
                q = np.sum(P_hat * (R_hat + mdp.gamma * np.max(q, axis=-1)), axis=-1)
                q[mdp.goal_states,:] = 0
            else: 
                q = np.sum(P_hat * (R_hat[:,:,np.newaxis] + mdp.gamma * np.max(q, axis=-1)), axis=-1)
                q[mdp.goal_states,:] = 0

            if np.max(np.abs(old_q-q)) < epsilon:
                break
                

        # Determine the best actions with bellman backups 
        if len(R_hat.shape) == 3:
            action_reward = np.sum(P_hat * (R_hat + mdp.gamma * np.max(q, axis=-1)), axis=-1)
        else:
            action_reward = np.sum(P_hat * (R_hat[:,:,np.newaxis] + mdp.gamma * np.max(q, axis=-1)), axis=-1)

        # Compute the optimal policy
        best_actions = (action_reward - np.max(action_reward, axis=1)[:,np.newaxis]) == 0
        policy = best_actions * (1 / np.sum( best_actions, axis=1)[:,np.newaxis])

        return policy, q

    # Compute the regret of the action with respect to the baseline policy
    @staticmethod
    def compute_regret(spibb, s, a):
        return np.maximum(spibb.V[s] - spibb.Q[s,a],0)
    
# Build environments, datasets, and baseline policies
class build:
    # Build a dataset of trajectories
    @staticmethod
    def build_D(mdp, policy, num_traj):

        trajectories = []

        for _ in range(num_traj):
            sars, g = mdp.sample_trajectory(policy)

            trajectories.append(sars)

        return trajectories


    # Generate a baseline policy
    @staticmethod
    def gen_baseline(mdp, targ_performance, dec=.1, epsilon = .001):

        opt_pol, q = utils.q_improvement(mdp)
        v_opt = np.sum(opt_pol * q, axis=-1)
        v_rand = utils.v_eval(mdp,mdp.rand_pol())

        pi_b = np.copy(opt_pol)
        pib_perf = build.compute_performance(mdp,pi_b, v_opt, v_rand)

        # Iteratively remove action weight from the best action
        while (pib_perf + epsilon) > targ_performance:

            actions = np.arange(mdp.A)
            flag = True
            i = 0

            while flag:

                s = np.random.choice(np.arange(mdp.S), 1)[0]
                i += 1

                # Find the best action in that state
                best_a = np.argmax(q[s,:])

                if pi_b[s,best_a] - dec >= 0:
                    flag = False

                if i > mdp.S:
                    print('Cannot find baseline')
                    return None

            pi_b[s,best_a] -= dec

            a = np.random.choice(actions[actions != best_a],1)[0]
            pi_b[s,a] += dec

            pib_perf = build.compute_performance(mdp, pi_b, v_opt, v_rand)
        
        return pi_b

    # Compute the normalized performance 
    # Where optimal policy = 1 and uniform random policy = 0
    @staticmethod
    def compute_performance(mdp, policy, v_opt, v_rand):
        return (utils.v_eval(mdp,policy)[mdp.init_state] - v_rand[mdp.init_state]) / \
                (v_opt[mdp.init_state] - v_rand[mdp.init_state])


    # Normalize the performance 
    # Where optimal policy = 1 and uniform random policy = 0
    @staticmethod
    def normalize_perf(mdp, perf, v_opt, v_rand):

        return (perf - v_rand[mdp.init_state]) / \
                (v_opt[mdp.init_state] - v_rand[mdp.init_state])


    # Return a variety of obstacle maps for GridWorld MDPs
    @staticmethod
    def obstacle_map(fname):

        if fname == '7x8':
            return     [
            "01000000",
            "01000000",
            "01001000",
            "00001000",
            "00001000",
            "00001000",
            "00001003"
                        ]
        elif fname == '7x8_lava':
            return     [
            "01000000",
            "01000000",
            "01001000",
            "00001000",
            "00001000",
            "00002000",
            "00002003"
                        ] 
        elif fname == '5x5_lava':
            return  [
            "00000",
            "00000",
            "00100",
            "02000",
            "20003",
                    ]
        elif fname == '5x5':
            return  [
            "00000",
            "00000",
            "00100",
            "00000",
            "00003",
                    ]
        
        elif fname == '25x25_lava':
            return [
                "0000000001000220000000000",
                "0010022001000000001000000",
                "0010000001001110002200001",
                "0010000001000000002200001",
                "0010000000000200000000001",
                "0010001020000200000000000",
                "0010001020000001111110000",
                "0000001000000000000000200",
                "0022001000222000000000200",
                "0000001000000000000000200",
                "0220000000002000011100000",
                "0000000000001100020000000",
                "0000112211002000000020003",
                "0000000000002000100020000",
                "0000200000000000100000000",
                "0000200000010000100022200",
                "0100000000010000000000000",
                "0100022200010000200010000",
                "0100022200010000200010000",
                "0100002100000000000010000",
                "0100000000111000000010000",
                "0000000000200000000010000",
                "0000100022000000000000000",
                "0000100022000211111000000",
                "0000100000000200000000000",
            ]
        
        elif fname == '25x25':
            return [
                "0000000001000000000000000",
                "0010000001000000000000000",
                "0010000001000000000000000",
                "0010000001000000000000000",
                "0010000000000000000000000",
                "0010001000000000000000000",
                "0010001000000001111110000",
                "0000001000000000000000000",
                "0000001000000000000000000",
                "0000001000000000000000000",
                "0000000000000000011100000",
                "0000000000000000000000000",
                "0000111111000000100000000",
                "0000000000000000100000000",
                "0000000000000000100000000",
                "0000000000010000100000000",
                "0100000000010000000000000",
                "0100000000010000000010000",
                "0100000000010000000010000",
                "0100000000000000000010000",
                "0100000000111000000010000",
                "0000000000000000000010000",
                "0000100000000000000000000",
                "0000100000000011111000000",
                "0000100000000000000000003",
            ]

# Compute SPIBB policy
class spibb:

        def __init__(self, mdp, N_wedge, pi_b):
            
            self.mdp = mdp
            self.N_wedge = N_wedge
            self.pi_b = pi_b

            self.spibb = np.copy(pi_b)

            self.Q = np.zeros((mdp.S, mdp.A))
            self.V = np.zeros(mdp.S)


        # Compute the SPIBB optimal policy
        def compute_spibb(self, P_hat, R_hat, N_D, epsilon=.01):
            self.Q = utils.approx_q_eval(self.mdp, P_hat, R_hat, self.pi_b)

            for _ in range(1000):
                old_Q = self.Q

                self.spibb = self.greedy_q_projection(self.Q, N_D)

                self.Q = utils.approx_q_eval(self.mdp, P_hat, R_hat, self.spibb)

                self.V = np.sum(self.spibb * self.Q,axis=-1)

                if np.max(np.abs(old_Q-self.Q)) < epsilon:
                    return self.spibb

        # Find the SPIBB optimal policy in the MLE MDP
        def greedy_q_projection(self, Q, N_D):

            spibb_pol = np.zeros((self.pi_b.shape[0],self.pi_b.shape[1]))

            for s in range(self.pi_b.shape[0]):
                for a in range(self.pi_b.shape[1]):

                    # Bootstrap with baseline policy in uncertain states
                    if N_D[s,a] < self.N_wedge:
                        spibb_pol[s,a] = self.pi_b[s,a]

                # Perform the greedy Q projection onto approx MDP
                safe_a = np.argwhere(N_D[s,:] >= self.N_wedge).flatten()
                if len(safe_a) != 0:
                    spibb_pol[s,safe_a[np.argmax(Q[s,safe_a])]] = np.sum(self.pi_b[s,safe_a])
                
            return spibb_pol

# Compute exploration policies
class explore:

    def __init__(self, mdp, pi_b):
        self.mdp = mdp
        self.pi_b = pi_b

    
    # Compute an approximate exploration policy based on a reward function
    # that rewards highest uncertainty and highest advantage
    def pi_ex(self, N_D, N_wedge, P_hat, R_hat):

        q_hat = utils.approx_q_eval(self.mdp, P_hat, R_hat, self.pi_b)
        v_hat = np.sum(self.pi_b * q_hat, axis=-1)

        # Compute approximate advantage
        a_hat = q_hat - v_hat[:,np.newaxis]
        a_hat = utils.min_max_norm(a_hat)

        # Compute the uncertainty over states
        u = np.maximum(0, 1 - (N_D/N_wedge) )
        
        opt_approx_pol, q = utils.approx_q_improvement(self.mdp, P_hat, u * a_hat, self.mdp.gamma)

        # Define approximate policy on unseen states
        unseen_states = np.sum(N_D ,axis=-1) == 0
        opt_approx_pol[unseen_states,:] = self.pi_b[unseen_states,:]

        return opt_approx_pol
    

    # Compute an approximate exploration policy based on a reward function
    # that rewards highest uncertainty
    def pi_ex_plan(self, N_D, N_wedge, P_hat, R_hat):
      
        q_hat = utils.approx_q_eval(self.mdp, P_hat, R_hat, self.pi_b)
        v_hat = np.sum(self.pi_b * q_hat, axis=-1)

        # Compute approximate advantage
        a_hat = q_hat - v_hat[:,np.newaxis]
        a_hat = utils.min_max_norm(a_hat)

        # Compute the uncertainty over states
        u = np.maximum(0, (1 - (N_D/N_wedge)))

        opt_approx_pol, q = utils.approx_q_improvement(self.mdp, P_hat, u[:,:,np.newaxis])

        # Define approximate policy on unseen states
        unseen_states = np.sum(N_D ,axis=-1) == 0
        opt_approx_pol[unseen_states,:] = self.pi_b[unseen_states,:]

        return opt_approx_pol

    # Compute an exploration policy that perturbs the baseline based on
    # the uncertainty and advantage
    def perturb_baseline(self, N_D, N_wedge, P_hat, R_hat):

        q_hat = utils.approx_q_reward_eval(self.mdp, P_hat, R_hat, self.pi_b)
        v_hat = np.sum(self.pi_b * q_hat, axis=-1)


        # Compute approximate advantage
        a_hat = q_hat - v_hat[:,np.newaxis]
        a_hat = utils.min_max_norm(a_hat)

        # Compute the uncertainty over states and scaling metric
        u = np.maximum(0, 1 - (N_D/N_wedge) )
        scale = np.exp(u) / np.sum(np.exp(u),axis=-1)[:,np.newaxis]
        
        # Scale the baseline policy
        pi_ex = self.pi_b * (scale / np.sum(scale,axis=-1)[:,np.newaxis])

        return  pi_ex / np.sum(pi_ex,axis=-1)[:,np.newaxis]
    

    # Compute an exploration policy that perturbs the baseline based on
    # the sum of uncertainty and advantage
    def perturb_baseline_sum(self, N_D, N_wedge, P_hat, R_hat):

        q_hat = utils.approx_q_eval(self.mdp, P_hat, R_hat, self.pi_b)
        v_hat = np.sum(self.pi_b * q_hat, axis=-1)

        # Compute approximate advantage
        a_hat = q_hat - v_hat[:,np.newaxis]
        a_hat = utils.min_max_norm(a_hat)

        # Compute the uncertainty over states and scaling factor
        u = np.maximum(0, 1 - (N_D/N_wedge) )
        extra_weight = np.sum(self.pi_b - self.pi_b * u,axis=-1)
        scale = np.exp(u) / np.sum(np.exp(u),axis=-1)[:,np.newaxis]
        
        # Return scaled baseline policy
        return scale * extra_weight[:,np.newaxis] + self.pi_b*u
    

    # Compute the exploration policy as the SPIBB policy
    def pi_spibb(self, N_D, N_wedge, P_hat, R_hat):
        return spibb(self.mdp, N_wedge, self.pi_b).run_spibb(P_hat, R_hat, N_D)
    

    # Explore using the optimal policy on the MLE MDP
    def pi_value_improve(self, N_D, N_wedge, P_hat, R_hat):

        pi_val, q = utils.approx_q_improvement(self.mdp, P_hat, R_hat)  

        # Define approximate policy on unseen states
        unseen_states = np.sum(N_D ,axis=-1) == 0
        pi_val[unseen_states,:] = self.pi_b[unseen_states,:]    
        return pi_val 
    
    
    # Explore using RMAX exploration
    def rmax(self, N_D, N_wedge, P_hat, R_hat):

        # Modify R_hat to fit RMAX criteria
        # Modify R_hat to fit RMAX criteria
        if len(R_hat.shape) == 3:
            R_hat = np.sum(R_hat * P_hat,axis=-1)
        R_hat[N_D < N_wedge] = self.mdp.rmax
        
        # Modify P_hat to fit RMAX criteria
        for s in range(N_D.shape[0]):
            for a in range(N_D.shape[1]):
                if N_D[s,a] < N_wedge:
                    P_hat[s,a,:] = 0
                    P_hat[s,a,s] = 1

        pi_rmax, q = utils.approx_q_improvement(self.mdp, P_hat, R_hat)
        
        return pi_rmax
    
    # Explore using RMAX exploration only on the dataset
    def rmax_dataset(self, N_D, N_wedge, P_hat, R_hat):

        # Modify R_hat to fit RMAX criteria
        if len(R_hat.shape) == 3:
            R_hat = np.sum(R_hat * P_hat,axis=-1)
        R_hat[(N_D < N_wedge) * (N_D > 0)] = self.mdp.rmax
        R_hat[N_D == 0] = self.mdp.rmin

        # Modify P_hat to fit RMAX criteria
        for s in range(N_D.shape[0]):
            for a in range(N_D.shape[1]):
                if N_D[s,a] < N_wedge:
                    P_hat[s,a,:] = 0
                    P_hat[s,a,s] = 1

        pi_rmax, q = utils.approx_q_improvement(self.mdp, P_hat, R_hat)

        return pi_rmax
    

    # 
    def uncertainty(self, N_D, N_wedge, P_hat, R_hat):

        R_tmp = np.zeros(P_hat.shape)

        for s in range(N_D.shape[0]):
            for a in range(N_D.shape[1]):
                if N_D[s,a] < N_wedge:
                    R_tmp[s,a,:] = (np.sum(N_D,axis=-1) < N_wedge) * self.mdp.rmax
                    R_tmp[s,a,self.mdp.goal_states] = self.mdp.rmax * 10 

                    P_hat[s,a,:] = 0
                    P_hat[s,a,s] = 1

                else:
                    R_tmp[s,a,:] = R_hat[s,a]

        pi_uncertain, q = utils.approx_q_improvement(self.mdp, P_hat, R_tmp)

        return pi_uncertain


    def uncertainty_v2(self, N_D, N_wedge, P_hat, R_hat):

        R_tmp = np.zeros(P_hat.shape)

        for s in range(N_D.shape[0]):
            for a in range(N_D.shape[1]):
                if N_D[s,a] < N_wedge:
                    R_tmp[s,a,:] = (np.sum(N_D,axis=-1) < N_wedge) * self.mdp.rmax
                    R_tmp[s,a,self.mdp.goal_states] = self.mdp.rmax * 10 

                    if N_D[s,a] == 0:
                        P_hat[s,a,:] = 0
                        P_hat[s,a,s] = 1
                else:
                    R_tmp[s,a,:] = R_hat[s,a]

        pi_uncertain, q = utils.approx_q_improvement(self.mdp, P_hat, R_tmp)

        return pi_uncertain

# General class for an MDP 
class mdp:

    # Set indices of trajectory tuples 
    S = 0
    A = 1
    R = 2
    S_P = 3

    def __init__(self, P, R, init_state, goal_states, gamma=.95, H=75):

        self.goal_states = goal_states
        self.init_state = init_state

        self.S = P.shape[0]
        self.A = P.shape[1]
        self.P = P
        self.R = R
        self.gamma = gamma

        self.H = H

        self.rmax = np.max(self.R)
        self.rmin = np.min(self.R)

    # Sample a trajectory given a policy
    def sample_trajectory(self, policy):

        sars = []

        self.curr_state = self.init_state

        g = 0
        i = 0

        while np.all(self.curr_state != self.goal_states):

            curr_action = np.random.choice(np.arange(self.A), 1, p=policy[self.curr_state,:])[0]

            next_state = np.random.choice(np.arange(self.S), 1, p=self.P[self.curr_state,curr_action,:])[0]

            if len(self.R.shape) == 3:
                reward = self.R[self.curr_state, curr_action, next_state]
            else: 
                reward = self.R[self.curr_state, curr_action]

            sars.append( [self.curr_state, curr_action, reward, next_state] )


            g += (self.gamma**i * reward)
            i += 1

            if len(sars) >= self.H:
                break

            self.curr_state = next_state
            
        return np.array(sars), g
    

    # Return the uniformly random policy 
    def rand_pol(self):
        return (1 / self.A) * np.ones((self.S,self.A))
    
    # Return a copy of the MDP
    @staticmethod
    def copy(orig_mdp):
        return mdp(np.copy(orig_mdp.P), np.copy(orig_mdp.R), \
                   np.copy(orig_mdp.init_state), np.copy(orig_mdp.goal_states))

# Defines a GridWorld from an obstacle map 
class grid_mdp:

    FREE = 0
    WALL = 1
    LAVA = 2
    GOAL = 3
    MOVES = {
            0: (-1, 0), #UP
            1: (1, 0),  #DOWN
            2: (0, -1), #LEFT
            3: (0, 1)   #RIGHT
        }

    # Set indices of trajectory tuples 
    S = 0
    A = 1
    R = 2
    S_P = 3


    def __init__(self, map, num_actions, init_state, gamma=.95, H=75):
        
        # Convert the state space to a numpy array and create action space
        self.state_space = np.asarray(map, dtype='c').astype('int')
        self.action_space = np.arange(num_actions)

        # Define the goal state and init state
        self.goal_states = mdp.to_s(self, np.argwhere(self.state_space==grid_mdp.GOAL).flatten())
        self.obstacle_states = mdp.to_s(self, np.argwhere(self.state_space==grid_mdp.LAVA).flatten())
        self.init_state = mdp.to_s(self, init_state)
        self.curr_state = self.init_state

        self.S = self.state_space.shape[0] * self.state_space.shape[1]
        self.A = num_actions
        self.P = self.build_stochastic_dynamics()
        self.R = self.build_stochastic_reward()
        self.gamma = gamma
        self.H = H

        self.rmax = np.max(self.R)
        self.rmin = np.min(self.R)

    # Convert a state tuple to an int
    @staticmethod
    def to_s(mdp, state):
        return state[0] * mdp.state_space.shape[1] + state[1]
    

    # Sample a trajectory on the MDP Dgiven a policy
    def sample_trajectory(self, policy):

        sars = []

        self.curr_state = self.init_state
        g = 0
        i = 0

        while np.all(self.curr_state != self.goal_states):

            curr_action = np.random.choice(np.arange(self.A), 1, p=policy[self.curr_state,:])[0]

            next_state = np.random.choice(np.arange(self.S), 1, p=self.P[self.curr_state,curr_action,:])[0]

            if len(self.R.shape) == 3:
                reward = self.R[self.curr_state, curr_action, next_state]
            else: 
                reward = self.R[self.curr_state, curr_action]

            sars.append( [self.curr_state, curr_action, reward, next_state] )

            g += self.gamma**i * reward
            i += 1

            if len(sars) >= self.H:
                break

            self.curr_state = next_state
            
        return np.array(sars), g
    
    # Return the uniformly random policy
    def rand_pol(self):
        return (1 / self.A) * np.ones((self.S,self.A))
        

    # Build stochastic gridworld dynamics 
    def build_stochastic_dynamics(self):

        x = self.state_space.shape[1]

        map = np.copy(self.state_space)
        map[map != grid_mdp.WALL] = grid_mdp.FREE
        map = map.flatten()

        dynamics = np.zeros((map.shape[0], self.A, map.shape[0]))

        for i in range(dynamics.shape[0]):
            
            # If in Wall, we stay there
            if map[i] == grid_mdp.WALL:
                dynamics[i,:,i] = 1

            else:
                
                # If a up move is valid
                if (i - x) >= 0 and map[i-x] != grid_mdp.WALL:
                    dynamics[i,0,i-x] = .75
                    dynamics[i,1,i-x] = .05
                    dynamics[i,2,i-x] = .1
                    dynamics[i,3,i-x] = .1
                else:
                    dynamics[i,0,i] += .75
                    dynamics[i,1,i] += .05
                    dynamics[i,2,i] += .1
                    dynamics[i,3,i] += .1

                # If a down move is valid
                if (i + x) < dynamics.shape[0] and map[i+x] != grid_mdp.WALL: 
                    dynamics[i,0,i+x] = .05
                    dynamics[i,1,i+x] = .75
                    dynamics[i,2,i+x] = .1
                    dynamics[i,3,i+x] = .1
                else:
                    dynamics[i,0,i] += .05
                    dynamics[i,1,i] += .75
                    dynamics[i,2,i] += .1
                    dynamics[i,3,i] += .1

                # If left move is valid
                if (i-1) % x != (x-1) and map[i-1] != grid_mdp.WALL:
                    dynamics[i,0,i-1] = .1
                    dynamics[i,1,i-1] = .1
                    dynamics[i,2,i-1] = .75
                    dynamics[i,3,i-1] = .05
                else:
                    dynamics[i,0,i] += .1
                    dynamics[i,1,i] += .1
                    dynamics[i,2,i] += .75
                    dynamics[i,3,i] += .05

                # If right move is valid 
                if (i+1) % x != 0 and map[i+1] != grid_mdp.WALL:
                    dynamics[i,0,i+1] = .1
                    dynamics[i,1,i+1] = .1
                    dynamics[i,2,i+1] = .05
                    dynamics[i,3,i+1] = .75
                else:
                    dynamics[i,0,i] += .1
                    dynamics[i,1,i] += .1
                    dynamics[i,2,i] += .05
                    dynamics[i,3,i] += .75


        return dynamics

    # Build deterministic dynamics model
    def build_dynamics(self):

        x = self.state_space.shape[1]

        map = np.copy(self.state_space)
        map[map != grid_mdp.WALL] = grid_mdp.FREE
        map = map.flatten()

        dynamics = np.zeros((map.shape[0], self.A, map.shape[0]))

        for i in range(dynamics.shape[0]):
            
            # If state is wall, do not move
            if map[i] == grid_mdp.WALL:
                dynamics[i,:,i] = 1

            else:
                
                # If up move is valid
                if (i - x) >= 0 and map[i-x] != grid_mdp.WALL:
                        dynamics[i,0,i-x] = 1
                else: 
                    dynamics[i,0,i] = 1
                
                # If down move is valid
                if (i + x) < dynamics.shape[0] and map[i+x] != grid_mdp.WALL:
                        dynamics[i,1,i+x] = 1
                else:
                    dynamics[i,1,i] = 1

                # If left move is valid
                if (i-1) % x != (x-1) and map[i-1] != grid_mdp.WALL:
                        dynamics[i,2,i-1] = 1
                else: 
                    dynamics[i,2,i] = 1

                # If right move is valid
                if (i+1) % x != 0 and map[i+1] != grid_mdp.WALL:
                        dynamics[i,3,i+1] = 1
                else: 
                    dynamics[i,3,i] = 1

        return dynamics
    
    # Build stochastic reward model 
    def build_stochastic_reward(self):

        rewards = np.zeros((self.S, self.A, self.S))
        
        # Set all goal states to +1
        rewards[:,:,self.goal_states] = 1

        # Set all obstacle states to -1
        if self.obstacle_states != None:
            rewards[:,:,self.obstacle_states] = -1

        return rewards

    # Build deterministc reward model
    def build_reward(self):

        map = self.state_space.flatten()
        
        rewards = np.zeros((self.S, self.A))

        for i in range(rewards.shape[0]):
            for j in range(rewards.shape[1]):

                next_states = np.argwhere(self.P[i,j] != 0)
                
                for ns in next_states:

                    if map[ns] == grid_mdp.GOAL:
                        rewards[i,j] = 1
                    elif map[ns] == grid_mdp.LAVA:
                        rewards[i,j] = -1 
        return rewards

    # Shuffle the initial and final states
    @staticmethod
    def change_init_reward(gmdp, goal_reward, obstacle_reward, movement_reward):

        grid = gmdp.P.shape[0]

        if grid == 625:
            map = build.obstacle_map('25x25_lava')
        elif grid == 56:
            map = build.obstacle_map('7x8_lava')
        elif grid == 25:
            map = build.obstacle_map('5x5_lava')

        map = np.asarray(map, dtype='c').astype('int')
        obstacle_map = map.flatten()

        init_state = np.random.randint(gmdp.S)
        
        while obstacle_map[init_state] == grid_mdp.LAVA or obstacle_map[init_state] == grid_mdp.WALL:
            init_state = np.random.randint(gmdp.S)
        
        goal_state = np.random.randint(gmdp.S)

        while obstacle_map[goal_state] == grid_mdp.LAVA or obstacle_map[goal_state] == grid_mdp.WALL:
            goal_state = np.random.randint(gmdp.S)

        gmdp.init_state = init_state
        gmdp.goal_states = goal_state
        gmdp.R = grid_mdp.update_reward(gmdp, map, goal_state, goal_reward, obstacle_reward, movement_reward)

        gmdp.rmax = np.max(gmdp.R)
        gmdp.rmin = np.min(gmdp.R)

    # Update stochastic reward model
    @staticmethod 
    def update_reward(mdp, map, goal_states, goal_reward, obstacle_reward, movement_reward):
        rewards = np.zeros((mdp.S, mdp.A, mdp.S)) + movement_reward

        obstacle_states = grid_mdp.static_to_s(map,np.argwhere(map==grid_mdp.LAVA).flatten())
        
        # Set all goal states to goal reward
        rewards[:,:,goal_states] = goal_reward

        # Set all lava states to obstacle reward
        if obstacle_states != None:
            rewards[:,:,obstacle_states] = obstacle_reward

        return rewards

    @staticmethod
    def static_to_s(map, state):
        return state[0] * map.shape[1] + state[1]

# Main algorithm class for SPI_HRL
class spi_hrl:

    def __init__(self, mdp, pi_b, D, N_wedge, N, regret=1, epsilon=.01):

        self.mdp = mdp
        self.epsilon = epsilon
        self.N = N
        self.pi_b = pi_b

        self.build_counts(D)

        self.spibb = spibb(self.mdp, N_wedge, pi_b)

        self.explore = explore(self.mdp, pi_b)

        self.regret_cap = regret

        self.horizon = int(np.mean([D[i].shape[0] for i in range(len(D))]))

        return 
    
    
    def run(self, j_max, method='perturb', pib_flag = False):

        j = 0
        policies = []
        counts = []
        self.spibb.compute_spibb(self.P_hat, self.R_hat, self.Nsa_D)

        policies.append(np.copy(self.spibb.spibb))
        counts.append(np.copy(self.Nsa_D))
        
        #while j < j_max:
        while np.max(np.fmax(0, 1-self.Nsa_D / self.spibb.N_wedge) * (self.Nsa_D > 0),axis=(-1,-2)) > 0:
        #while np.max(np.fmax(0, 1-self.Nsa_D / self.spibb.N_wedge) * (self.Nsa_D > 0),axis=(-1,-2)) > 0:
        #while True:

            for _ in range(self.N):

                if pib_flag:
                    trajectory, g = self.mdp.sample_trajectory(self.pi_b)

                else:
                    if method == 'pi_ex':
                        pi_ex  = self.explore.pi_ex(self.Nsa_D, self.spibb.N_wedge, self.P_hat, self.R_hat)
                    elif method == 'pi_ex_plan':
                        pi_ex  = self.explore.pi_ex_plan(self.Nsa_D, self.spibb.N_wedge, self.P_hat, self.R_hat)
                    elif method == 'spibb':
                        pi_ex  = self.explore.pi_spibb(self.Nsa_D, self.spibb.N_wedge, self.P_hat, self.R_hat)
                    elif method == 'perturb':
                        pi_ex = self.explore.perturb_baseline(self.Nsa_D, self.spibb.N_wedge, self.P_hat, self.R_hat)
                    elif method == 'perturb_sum':
                        pi_ex = self.explore.perturb_baseline_sum(self.Nsa_D, self.spibb.N_wedge, self.P_hat, self.R_hat)
                    elif method == 'val_ex':
                        pi_ex = self.explore.pi_value_improve(self.Nsa_D, self.spibb.N_wedge, self.P_hat, self.R_hat)
                    elif method == 'rmax':
                        pi_ex = self.explore.rmax(self.Nsa_D, self.spibb.N_wedge, self.P_hat, self.R_hat)
                    elif method == 'rmax_dataset':
                        pi_ex = self.explore.rmax_dataset(self.Nsa_D, self.spibb.N_wedge, self.P_hat, self.R_hat)
                    elif method == 'uncertainty':
                        pi_ex = self.explore.uncertainty(self.Nsa_D, self.spibb.N_wedge, self.P_hat, self.R_hat)
                    elif method == 'uncertainty_v2':
                        pi_ex = self.explore.uncertainty_v2(self.Nsa_D, self.spibb.N_wedge, self.P_hat, self.R_hat)

                    trajectory, g = self.sample_trajectory(pi_ex)

                self.update_counts(trajectory)

           
            # Find the SPIBB policy
            self.spibb.compute_spibb(self.P_hat, self.R_hat, self.Nsa_D)
            policies.append(np.copy(self.spibb.spibb))
            counts.append(np.copy(self.Nsa_D))

            j+=1
            if j > 250:
                break

        return policies, counts
    

    # Sample a trajectory given a policy
    def sample_trajectory(self, policy):
        sars = []

        self.mdp.curr_state = self.mdp.init_state

        g = 0
        i = 0
        regret = 0

        while np.all(self.mdp.curr_state != self.mdp.goal_states):

            # If below regret cap, use exploratory policy
            # Otherwise default to the baseline
            if  regret < self.regret_cap:
            
                # If in a seen state-action, compute regret
                if np.sum(self.Nsa_D[self.mdp.curr_state,:]) > 0:
                    curr_action = np.random.choice(np.arange(self.mdp.A), 1, p=policy[self.mdp.curr_state,:])[0]
                    regret += utils.compute_regret(self.spibb, self.mdp.curr_state, curr_action)
                else: 
                    curr_action = np.random.choice(np.arange(self.mdp.A), 1, p=self.pi_b[self.mdp.curr_state,:])[0]
            else:
                curr_action = np.random.choice(np.arange(self.mdp.A), 1, p=self.pi_b[self.mdp.curr_state,:])[0]

            next_state = np.random.choice(np.arange(self.mdp.S), 1, p=self.mdp.P[self.mdp.curr_state,curr_action,:])[0]

            # Get Reward
            if len(self.mdp.R.shape) == 3:
                reward = self.mdp.R[self.mdp.curr_state, curr_action, next_state]
            else: 
                reward = self.mdp.R[self.mdp.curr_state, curr_action]

            sars.append( [self.mdp.curr_state, curr_action, reward, next_state] )

            # Compute the return
            g += (self.mdp.gamma**i * reward)
            i += 1
            
            if len(sars) >= self.mdp.H or len(sars) >= self.horizon:
                break

            # Update state
            self.mdp.curr_state = next_state
            
        return np.array(sars), g
    
    def update_counts(self, trajectory):
            for sarcs in trajectory:

                self.Nsas_D[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A]), int(sarcs[grid_mdp.S_P])] += 1
                self.Nsa_D[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])] +=1

                if len(self.mdp.R.shape) == 3:
                    self.R_hat[int(sarcs[grid_mdp.S]),int(sarcs[grid_mdp.A]),int(sarcs[grid_mdp.S_P])] += -(self.R_hat[int(sarcs[grid_mdp.S]),int(sarcs[grid_mdp.A]),int(sarcs[grid_mdp.S_P])]\
                                                        / self.Nsas_D[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A]),int(sarcs[grid_mdp.S_P])]) + \
                                                        (sarcs[grid_mdp.R] / self.Nsas_D[int(sarcs[grid_mdp.S]),int(sarcs[grid_mdp.A]),int(sarcs[grid_mdp.S_P])])
                else:
                    self.R_hat[int(sarcs[grid_mdp.S]),int(sarcs[grid_mdp.A])] += -(self.R_hat[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])]\
                                                        / self.Nsa_D[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])]) + \
                                                        (sarcs[grid_mdp.R] / self.Nsa_D[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])])
                
            self.P_hat = self.Nsas_D / np.where(self.Nsa_D > 0, self.Nsa_D, 1)[:,:,np.newaxis]

    def build_counts(self, D):
            self.Nsas_D = np.zeros((self.mdp.S,self.mdp.A,self.mdp.S))
            self.Nsa_D = np.zeros((self.mdp.S,self.mdp.A))
            self.R_hat = np.zeros(self.mdp.R.shape)

            for traj in D:
                for sarcs in traj:
                    self.Nsas_D[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A]), int(sarcs[grid_mdp.S_P])] += 1
                    self.Nsa_D[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])] +=1

                    if len(self.mdp.R.shape) == 3:
                        self.R_hat[int(sarcs[grid_mdp.S]),int(sarcs[grid_mdp.A]),int(sarcs[grid_mdp.S_P])] += -(self.R_hat[int(sarcs[grid_mdp.S]),int(sarcs[grid_mdp.A]),int(sarcs[grid_mdp.S_P])]\
                                                            / self.Nsas_D[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A]),int(sarcs[grid_mdp.S_P])]) + \
                                                            (sarcs[grid_mdp.R] / self.Nsas_D[int(sarcs[grid_mdp.S]),int(sarcs[grid_mdp.A]),int(sarcs[grid_mdp.S_P])])
                    else:
                        self.R_hat[int(sarcs[grid_mdp.S]),int(sarcs[grid_mdp.A])] += -(self.R_hat[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])]\
                                                            / self.Nsa_D[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])]) + \
                                                            (sarcs[grid_mdp.R] / self.Nsa_D[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])])
            
            self.P_hat = self.Nsas_D / np.where(self.Nsa_D > 0, self.Nsa_D, 1)[:,:,np.newaxis]
    
# Main alg class for RMAX agent
class RMAXAgent:

    def __init__(self, mdp, D=[], m=20):
        self.name = 'rmax'
        self.mdp = mdp
        self.m = m

        self.Q = np.zeros((self.mdp.S,self.mdp.A)) + self.mdp.rmax
        self.r = np.zeros(self.mdp.R.shape)
        self.n_sa = np.zeros((self.mdp.S,self.mdp.A))
        self.n_sas = np.zeros((self.mdp.S,self.mdp.A,self.mdp.S))

        self.R_hat = np.zeros(self.mdp.R.shape)
        self.P_hat = np.zeros(self.mdp.P.shape)

        self.NSA = np.zeros((self.mdp.S,self.mdp.A))
        self.NSAS = np.zeros((self.mdp.S,self.mdp.A,self.mdp.S))

        for trajectory in D:
            self.update_counts(trajectory)
    
    def run(self):

        # Compute the RMAX policy
        self.Q[self.n_sa < self.m] = self.mdp.rmax / (1-self.mdp.gamma)

        best_actions = (self.Q - np.max(self.Q, axis=1)[:,np.newaxis]) == 0
        rmax_pol = best_actions * (1 / np.sum( best_actions, axis=1)[:,np.newaxis])

        traj, returns = self.mdp.sample_trajectory(rmax_pol)

        self.update_counts(traj)

        # Recompute Q and the opitimal policy
        pi_star, self.Q = utils.approx_q_improvement(self.mdp, self.P_hat, self.R_hat)

        return {'nsas':np.copy(self.NSAS), 'nsa':np.copy(self.NSA), 'phat':np.copy(self.P_hat), 'rhat':np.copy(self.R_hat), 'pistar':np.copy(pi_star)}
         
    def update_counts(self, trajectory):

        for sarcs in trajectory:
            
            # Update model counts
            if self.n_sa[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])] < self.m:

                self.n_sas[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A]), int(sarcs[grid_mdp.S_P])] += 1
                self.n_sa[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])] +=1

                if self.r.shape == 3:
                    self.r[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A]), int(sarcs[grid_mdp.S_P])] += sarcs[grid_mdp.R]
                else: 
                    self.r[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])] += sarcs[grid_mdp.R]
            
            # Update total seen counts
            self.NSAS[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A]), int(sarcs[grid_mdp.S_P])] += 1
            self.NSA[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])] +=1

        # Update transitions and reward 
        self.P_hat = self.n_sas / np.where(self.n_sa > 0, self.n_sa, 1)[:,:,np.newaxis]

        if len(self.r.shape) == 3:
            self.R_hat = self.r / np.where(self.n_sas > 0, self.n_sas, 1)
        else:
            self.R_hat = self.r / np.where(self.n_sa > 0, self.n_sa, 1)

# Main alg class for UCBVI agent
class UCBVIAgent:

    def __init__(self, mdp, D=[], delta=.95,K=50):
        self.name = 'ucbvi'
        self.mdp = mdp
        self.delta = delta

        self.K = K
        self.T = self.K * self.mdp.H

        self.Q = np.zeros((self.mdp.S,self.mdp.A))
        self.n_sa = np.zeros((self.mdp.S,self.mdp.A))
        self.n_sas = np.zeros((self.mdp.S,self.mdp.A,self.mdp.S))
        self.r = np.zeros(self.mdp.R.shape)

        self.R_hat = np.zeros(self.mdp.R.shape)
        self.P_hat = np.zeros(self.mdp.P.shape)

        self.D = D

        for trajectory in self.D:
            self.update_counts(trajectory)
        
    def run(self):

        b = self.bonus()

        # Compute the UCBVI policy
        V = np.zeros((self.mdp.H+1,self.mdp.S))
        Q = np.zeros((self.mdp.H+1,self.mdp.S,self.mdp.A))
        for h in range(self.mdp.H)[::-1]:
            for s in range(self.mdp.S):
                for a in range(self.mdp.A):
                    if self.n_sa[s,a] == 0:
                        Q[h,s,a] = self.mdp.H
                    else:
                        if len(self.R_hat.shape) == 3:
                            Q[h,s,a] = np.min( [Q[h,s,a], self.mdp.H, np.sum(self.P_hat[s,a]*self.R_hat[s,a],axis=-1) + \
                                            np.sum(self.P_hat[s,a,:]*V[h+1],axis=-1) + b[s,a]])
                        else:
                            Q[h,s,a] = np.min( [Q[h,s,a], self.mdp.H, self.R_hat[s,a] + \
                                            np.sum(self.P_hat[s,a,:]*V[h+1],axis=-1) + b[s,a]])
        self.Q = Q[0]

        best_actions = (self.Q - np.max(self.Q, axis=1)[:,np.newaxis]) == 0
        ucbvi_pol = best_actions * (1 / np.sum( best_actions, axis=1)[:,np.newaxis])
        
        traj, returns = self.mdp.sample_trajectory(ucbvi_pol)

        self.update_counts(traj)

        # Recompute Q and the opitimal policy
        pi_star, self.Q = utils.approx_q_improvement(self.mdp, self.P_hat, self.R_hat)

        return {'nsas':np.copy(self.n_sas), 'nsa':np.copy(self.n_sa), 'phat':np.copy(self.P_hat), 'rhat':np.copy(self.R_hat), 'pistar':np.copy(pi_star)}
         
    # Update counts from trajectory
    def update_counts(self, trajectory):

        for sarcs in trajectory:
            self.n_sas[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A]), int(sarcs[grid_mdp.S_P])] += 1
            self.n_sa[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])] +=1

            if self.r.shape == 3:
                self.r[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A]), int(sarcs[grid_mdp.S_P])] += sarcs[grid_mdp.R]
            else: 
                self.r[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])] += sarcs[grid_mdp.R]
            
        # Update transitions and reward 
        self.P_hat = self.n_sas / np.where(self.n_sa > 0, self.n_sa, 1)[:,:,np.newaxis]

        if len(self.r.shape) == 3:
            self.R_hat = self.r / np.where(self.n_sas > 0, self.n_sas, 1)
        else:
            self.R_hat = self.r / np.where(self.n_sa > 0, self.n_sa, 1)

    def bonus(self):
        return ( (7*self.mdp.H) / np.sqrt(np.where(self.n_sa > 0, self.n_sa, 1)) ) * np.log( (5*self.mdp.S*self.mdp.A*self.T) / self.delta)

# Main Alg class for Reward Free Non Reactive Policy Design agent
class RFNPDAgent:
    def __init__(self, mdp, D=[], theta=5, delta=.95, K=50):
        self.name = 'rfnpd'
        self.mdp = mdp
        self.delta = delta
        self.theta = theta
        self.K = K

        self.Q = np.zeros((self.mdp.S,self.mdp.A))
        self.n_sa = np.zeros((self.mdp.S,self.mdp.A))
        self.n_sas = np.zeros((self.mdp.S,self.mdp.A,self.mdp.S))
        self.r = np.zeros(self.mdp.R.shape)

        self.R_hat = np.zeros(self.mdp.R.shape)
        self.P_hat = np.zeros(self.mdp.P.shape)
        
        self.NSA = np.zeros((self.mdp.S+1,self.mdp.A))
        self.NSAS = np.zeros((self.mdp.S+1,self.mdp.A,self.mdp.S+1))

        self.D = D
        for trajectory in self.D:
            self.update_counts(trajectory)
            self.update_virtual_counts(trajectory)

    def run(self):
        # Compute the RFNRP policy
        rfnrp_pol = self.compute_nrp()

        traj, returns = self.mdp.sample_trajectory(rfnrp_pol)

        self.update_counts(traj)

        # Recompute Q and the opitimal policy
        pi_star, self.Q = utils.approx_q_improvement(self.mdp, self.P_hat, self.R_hat)

        return {'nsas':np.copy(self.n_sas), 'nsa':np.copy(self.n_sa), 'phat':np.copy(self.P_hat), 'rhat':np.copy(self.R_hat), 'pistar':np.copy(pi_star)}

    

    # Compute the non-reactive exploration policy
    def compute_nrp(self):

        sparse_mdp = self.build_sparse_mdp()

        U = np.zeros((self.K,self.mdp.H+1,self.mdp.S,self.mdp.A))
        pi = np.zeros((self.K,self.mdp.H,self.mdp.S+1,self.mdp.A))
        for k in range(self.K):
            for h in range(self.mdp.H)[::-1]:
                # Compute the uncertainty
                b = self.bonus()
                U[k,h] = self.mdp.H * np.minimum(1, b) + np.sum(sparse_mdp.P[:-1,:,:-1] * np.max(U[k,h+1],axis=-1),axis=-1)

                # Compute the policy
                a = np.argmax(U[k,h],axis=-1)
                for s in range(a.shape[0]):
                    pi[k,h,s,a[s]] = 1
                pi[k,h,-1,0] = 1

            # Sample trajectory, update counts
            traj, _ = sparse_mdp.sample_trajectory(pi[k,0])
            self.update_virtual_counts(traj)

        # Take the uniform mixture over the stationary policies
        return np.mean(pi[:,0],axis=0)[:-1]

    def update_counts(self, trajectory):
        for sarcs in trajectory:
            self.n_sas[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A]), int(sarcs[grid_mdp.S_P])] += 1
            self.n_sa[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])] +=1

            if self.r.shape == 3:
                self.r[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A]), int(sarcs[grid_mdp.S_P])] += sarcs[grid_mdp.R]
            else: 
                self.r[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])] += sarcs[grid_mdp.R]
            
        # Update transitions and reward 
        self.P_hat = self.n_sas / np.where(self.n_sa > 0, self.n_sa, 1)[:,:,np.newaxis]

        if len(self.r.shape) == 3:
            self.R_hat = self.r / np.where(self.n_sas > 0, self.n_sas, 1)
        else:
            self.R_hat = self.r / np.where(self.n_sa > 0, self.n_sa, 1)

    # Update counts for sparse MDP
    def update_virtual_counts(self, trajectory):
        for sarcs in trajectory:
            self.NSAS[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A]), int(sarcs[grid_mdp.S_P])] += 1
            self.NSA[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])] +=1

    # Build the Sparsified MDP
    def build_sparse_mdp(self):

        # Build sparsified dynamcis
        P = np.zeros((self.mdp.S+1, self.mdp.A, self.mdp.S+1))
        P[:-1,:,:-1] = np.where(self.n_sas >= self.theta, self.P_hat, 0)
        P[:-1,:,-1] = 1 - np.sum(P[:-1,:,:-1],axis=-1)
        P[np.sum(P,axis=-1)==0,-1] = 1

        for sa in np.argwhere(P < 0):
            P[sa[0],sa[1],np.argmax(P[sa[0],sa[1]] > 0)] += P[sa[0],sa[1],sa[2]]
            P[sa[0],sa[1],sa[2]] = 0



        # Build sparsified reward
        if len(self.mdp.R.shape) == 3:
            R = np.zeros((self.mdp.S+1, self.mdp.A, self.mdp.S+1))
            R[:-1,:,:-1] = self.R_hat
        else:
            R = np.zeros((self.mdp.S+1, self.mdp.A))
            R[:-1,:] = self.R_hat

        return mdp(P,R, np.copy(self.mdp.init_state), np.copy(self.mdp.goal_states))

    def bonus(self):
        return (self.mdp.H / self.NSA[:-1]) * (np.log((6*self.mdp.H*self.mdp.S*self.mdp.A)/self.delta) \
                                + self.mdp.S*np.log(np.e*(1+self.NSA[:-1]/self.mdp.S)))

class bonusRMAXAgent:

    def __init__(self, mdp, D=[], m=20):
        self.name = 'bonusrmax'
        self.mdp = mdp
        self.m = m

        self.Q = np.zeros((self.mdp.S,self.mdp.A)) + self.mdp.rmax
        self.r = np.zeros(self.mdp.R.shape)
        self.n_sa = np.zeros((self.mdp.S,self.mdp.A))
        self.n_sas = np.zeros((self.mdp.S,self.mdp.A,self.mdp.S))

        self.R_hat = np.zeros(self.mdp.R.shape)
        self.P_hat = np.zeros(self.mdp.P.shape)

        for trajectory in D:
            self.update_counts(trajectory)
    
    def run(self):

        # Compute the uncertainty over states
        u = np.maximum(0, 1 - (self.n_sa/self.m) )

        # Compute the RMAX bonus policy based on uncertainty
        self.Q[self.n_sa < self.m] = self.mdp.rmax / (1-self.mdp.gamma) + (self.Q*u)[self.n_sa < self.m] / (1-self.mdp.gamma)

        best_actions = (self.Q - np.max(self.Q, axis=1)[:,np.newaxis]) == 0
        rmax_pol = best_actions * (1 / np.sum( best_actions, axis=1)[:,np.newaxis])

        traj, returns = self.mdp.sample_trajectory(rmax_pol)

        self.update_counts(traj)

        # Recompute Q and the opitimal policy
        pi_star, self.Q = utils.approx_q_improvement(self.mdp, self.P_hat, self.R_hat)

        return {'nsas':np.copy(self.n_sas), 'nsa':np.copy(self.n_sa), 'phat':np.copy(self.P_hat), 'rhat':np.copy(self.R_hat), 'pistar':np.copy(pi_star)}
         

    def update_counts(self, trajectory):

        for sarcs in trajectory:
            
            self.n_sas[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A]), int(sarcs[grid_mdp.S_P])] += 1
            self.n_sa[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])] +=1

            if self.r.shape == 3:
                self.r[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A]), int(sarcs[grid_mdp.S_P])] += sarcs[grid_mdp.R]
            else: 
                self.r[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])] += sarcs[grid_mdp.R]
            
        # Update transitions and reward 
        self.P_hat = self.n_sas / np.where(self.n_sa > 0, self.n_sa, 1)[:,:,np.newaxis]

        if len(self.r.shape) == 3:
            self.R_hat = self.r / np.where(self.n_sas > 0, self.n_sas, 1)
        else:
            self.R_hat = self.r / np.where(self.n_sa > 0, self.n_sa, 1)


class VIECAgent:
    def __init__(self, mdp, D=[], m=20, epsilon=.01):
        self.name = 'viec'
        self.mdp = mdp
        self.m = m

        self.Q = np.zeros((self.mdp.S,self.mdp.A)) + self.mdp.rmax
        self.r = np.zeros(self.mdp.R.shape)
        self.n_sa = np.zeros((self.mdp.S,self.mdp.A))
        self.n_sas = np.zeros((self.mdp.S,self.mdp.A,self.mdp.S))

        self.R_hat = np.zeros(self.mdp.R.shape)
        self.P_hat = np.zeros(self.mdp.P.shape)

        self.D = D
        self.epsilon = epsilon

        for trajectory in self.D:
            self.update_counts(trajectory)
    
    def run(self):

        viec_pol = self.compute_exploration()
      
        traj, returns = self.mdp.sample_trajectory(viec_pol)

        self.update_counts(traj)

        # Recompute Q and the opitimal policy
        pi_star, self.Q = utils.approx_q_improvement(self.mdp, self.P_hat, self.R_hat)

        return {'nsas':np.copy(self.NSAS), 'nsa':np.copy(self.NSA), 'phat':np.copy(self.P_hat), 'rhat':np.copy(self.R_hat), 'pistar':np.copy(pi_star)}
    

    # Update state-action counts
    def update_counts(self, trajectory):

        for sarcs in trajectory:
            
            # Update model counts
            self.n_sas[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A]), int(sarcs[grid_mdp.S_P])] += 1
            self.n_sa[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])] +=1

            if self.r.shape == 3:
                self.r[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A]), int(sarcs[grid_mdp.S_P])] += sarcs[grid_mdp.R]
            else: 
                self.r[int(sarcs[grid_mdp.S]), int(sarcs[grid_mdp.A])] += sarcs[grid_mdp.R]
            
        # Update transitions and reward 
        self.P_hat = self.n_sas / np.where(self.n_sa > 0, self.n_sa, 1)[:,:,np.newaxis]

        if len(self.r.shape) == 3:
            self.R_hat = self.r / np.where(self.n_sas > 0, self.n_sas, 1)
        else:
            self.R_hat = self.r / np.where(self.n_sa > 0, self.n_sa, 1)
    
    def compute_exploration(self):
        self.demand = np.maximum(0, self.m-self.n_sa)

        demand_mats = self.enumerate_demand()

        U = np.zeros((demand_mats.shape[0], self.mdp.S))
        C = np.zeros(demand_mats.shape)

        for i in range(demand_mats.shape[0]-1):
            thresh = 0
            while True:
                d = demand_mats[i]
                for s in range(self.mdp.S):
                    for a in range(self.mdp.A):
                        k = np.argwhere(self.H(d,s,a) == demand_mats,axis=0).flatten()
                        c = 1 + np.sum( P[s,a,:] * U[k])
                        thresh = np.maximum( thresh, np.abs(C[i,s,a]-c) )
                        C[i,s,a] = c
                    U[i,s] = np.min(C[i,s])
                if thresh < self.epsilon:
                    break
        
        # Compute the deterministic policy
        pi = np.zeros((self.mdp.S,self.mdp.A))
        a = np.argmin(C[-1],axis=-1)
        for s in range(a.shape[0]):
            pi[s,a[s]] = 1

        return pi
    # H decrementor function
    def H(self, d, s, a):
        return (d[s,a] - 1) if d[s,a] > 0 else d
    
    # Enumerate all possible demand matrices
    def enumerate_demand(self):
        demand_mats = np.zeros((int(np.prod(self.demand+1)),self.mdp.S,self.mdp.A))

        j = 0
        l = 0
        for s in range(self.mdp.S)[::-1]:
            for a in range(self.mdp.A)[::-1]:
                for k in range(self.demand[s,a]+1):
                    if self.demand[s,a] < 1:
                        break
                    if len(demand_mats[1:l]) < 1:
                        demand_mats[j,s,a] += k
                        j+=1
                    else:
                        if k == 0:
                            continue
                        for dm in demand_mats[:l]:
                            demand_mats[j,s,a] += k
                            demand_mats[j] += np.copy(dm)
                            j+=1
                l=j
        return demand_mats

# Generate trials and plot data
class plot:

    # Generate trials 
    @staticmethod
    def gen_trials(path, fname, ename, method, mdp, pi_b, baseline_perf, d_size, n_wedge, N, j_max, reps, regret):
        opt_pol, v = utils.value_improvement(mdp)
        v_rand = utils.v_eval(mdp,mdp.rand_pol())

        hrl_res = []
        pib_res = []
        pib_counts_res = []
        hrl_counts_res = []
        pib_pols = []
        hrl_pols = []

        for k in range(reps):
            print(f'Iteration: {k}')
            

            D = build.build_D(mdp, pi_b, d_size)

            spi = spi_hrl(mdp, pi_b, D, n_wedge, N, regret=regret)
            hrl_spibb, pi_ex_counts = spi.run(j_max, method, pib_flag=False)

            spi_pib = spi_hrl(mdp, pi_b, D, n_wedge, N, regret=regret)
            spibb_spibb, pi_b_counts = spi_pib.run(j_max, method, pib_flag=True)

            hrl_res.append([build.compute_performance(mdp,hrl_spibb[i],v,v_rand) for i in range(len(hrl_spibb))])
            pib_res.append([build.compute_performance(mdp,spibb_spibb[i],v,v_rand) for i in range(len(spibb_spibb))])
            pib_counts_res.append(pi_b_counts)
            hrl_counts_res.append(pi_ex_counts)
            hrl_pols.append(hrl_spibb)
            pib_pols.append(spibb_spibb)

        max_hrl = 0
        max_pib = 0
        for k in range(reps):
            if len(hrl_res[k]) > max_hrl:
                max_hrl = len(hrl_res[k])
            if len(pib_res[k]) > max_pib:
                max_pib = len(pib_res[k])

        for k in range(reps):
            for j in range(max_hrl - len(hrl_res[k])):
                hrl_res[k].append(hrl_res[k][-1])
                hrl_counts_res[k].append(hrl_counts_res[k][-1])
                hrl_pols[k].append(hrl_pols[k][-1])

            for j in range(max_pib - len(pib_res[k])):
                pib_res[k].append(pib_res[k][-1])
                pib_counts_res[k].append(pib_counts_res[k][-1])
                pib_pols[k].append(pib_pols[k][-1])

        # Make the path if none exists
        Path(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}').mkdir(parents=True,exist_ok=True)

        np.save(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_piHRL.npy', hrl_res)
        np.save(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_piSPIBB.npy',pib_res)
        np.save(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_piex_counts.npy',hrl_counts_res)
        np.save(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_pib_counts.npy',pib_counts_res)
        np.save(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_hrl_pols.npy',hrl_pols)
        np.save(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_spibb_pols.npy',pib_pols)


    # Plot the performance of the SPIBB policies
    @staticmethod
    def plot_perf(path, fname, ename, method, mdp, pi_b, baseline_perf, d_size, n_wedge, N, j_max, reps, regret):
        opt_pol, v = utils.value_improvement(mdp)
        v_rand = utils.v_eval(mdp,mdp.rand_pol())
        try:
            hrl_res = np.load(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_piHRL.npy')
            spibb_res = np.load(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_piSPIBB.npy')
            hrl_counts_res = np.load(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_piex_counts.npy')
            pib_counts_res = np.load(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_pib_counts.npy')
            hrl_pols = np.load(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_hrl_pols.npy')
            spibb_pols = np.load(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_spibb_pols.npy')
        except:
            return

        plt.figure()
        j = np.maximum(spibb_res.shape[1],hrl_res.shape[1])
        plt.plot(np.mean(hrl_res,axis=0),alpha=.7, label='SPIBB with D=D_off+pi_ex',marker='o')
        plt.plot(np.mean(spibb_res,axis=0),alpha=.7, label='SPIBB D=D_off+pi_b', marker='s')
        plt.plot(np.arange(j),np.tile(build.compute_performance(mdp,opt_pol,v,v_rand),j),c='k',linestyle='dotted', label='Optimal Performance')
        plt.plot(np.arange(j),np.tile(build.compute_performance(mdp,pi_b,v,v_rand),j),c='k',linestyle='dashed', label='Baseline Performance')
        plt.legend(loc='lower right')

        plt.title(f'Comparison of SPIBB Policy Expected Returns on Online+Offline Datasets')
        plt.ylabel('Average Performance')
        plt.xlabel('Number of Trajectories')
        plt.xticks(np.arange(j, step=10), labels=np.arange(j,step=10)*N+d_size,rotation=45)

        Path(f'{path}mdp_data/figs/{fname}/{ename}/{method}').mkdir(parents=True,exist_ok=True)
        plt.savefig(f'{path}mdp_data/figs/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_perf.png')

    # Plot the number of state-action pairs seen by the agent over time
    @staticmethod
    def plot_counts_graph(path, fname, ename, method, mdp, pi_b, baseline_perf, d_size, n_wedge, N, j_max, reps, regret):
        v_range = [0,1]

        ndsa_hrl = np.load(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_piex_counts.npy')
        ndsa_pib = np.load(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_pib_counts.npy')

        pib_new = []
        hrl_new = []

        for i in range(ndsa_hrl.shape[1]):
            hrl_new.append(np.sum(ndsa_hrl[:,i] > 0, (-1, -2)))
            
        for j in range(ndsa_pib.shape[1]):
            pib_new.append(np.sum(ndsa_pib[:,j] > 0, (-1, -2)))

        hrl_new = np.array(hrl_new)
        pib_new = np.array(pib_new)
        plt.figure()
        j = np.maximum(ndsa_hrl.shape[1],ndsa_pib.shape[1])
        plt.plot(np.mean(hrl_new,axis=-1),alpha=.7, label='D=D+pi_ex',marker='o')
        plt.plot(np.mean(pib_new,axis=-1),alpha=.7, label='D=D+pi_b', marker='s')

        plt.legend(loc='lower right')

        plt.title(f'New (s,a) pairs visited comparing exploration methods')
        plt.ylabel('Number of (s,a) pairs')
        plt.xlabel('Number of Trajectories')
        plt.xticks(np.arange(j,step=10), labels=np.arange(j,step=10)*N+d_size,rotation=45)

        Path(f'{path}mdp_data/figs/{fname}/{ename}/{method}').mkdir(parents=True,exist_ok=True)
        plt.savefig(f'{path}mdp_data/figs/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_counts.png')


    # Plot the uncertainty over seen state-action pairs
    @staticmethod
    def plot_uncertainty_graph(path, fname, ename, method, mdp, pi_b, baseline_perf, d_size, n_wedge, N, j_max, reps,regret):

        ndsa_hrl = np.load(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_piex_counts.npy')
        ndsa_pib = np.load(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_pib_counts.npy')

        pib_new = []
        hrl_new = []

        for i in range(ndsa_hrl.shape[1]):
            hrl_new.append(np.max(np.fmax(0, 1-ndsa_hrl[:,i]/n_wedge) * (ndsa_hrl[:,i] > 0),axis=(-1,-2)))
        for j in range(ndsa_pib.shape[1]):
            pib_new.append(np.max(np.fmax(0, 1-ndsa_pib[:,j]/n_wedge) * (ndsa_pib[:,j] > 0),axis=(-1,-2)))

        hrl_new = np.array(hrl_new)
        pib_new = np.array(pib_new)
        plt.figure()
        j = np.maximum(ndsa_hrl.shape[1],ndsa_pib.shape[1])
        plt.plot(np.mean(hrl_new,axis=-1),alpha=.7, label='D=D+pi_ex',marker='o')
        plt.plot(np.mean(pib_new,axis=-1),alpha=.7, label='D=D+pi_b', marker='s')


        plt.legend(loc='lower right')

        plt.title(f'Uncertainty Reduction Under Exploration methods')
        plt.ylabel('Uncertainty')
        plt.xlabel('Number of Trajectories')
        plt.xticks(np.arange(j,step=10), labels=np.arange(j,step=10)*N+d_size,rotation=45)

        Path(f'{path}mdp_data/figs/{fname}/{ename}/{method}').mkdir(parents=True,exist_ok=True)
        plt.savefig(f'{path}mdp_data/figs/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_uncertainty.png')

    # Plot the number of certain state-action pairs
    @staticmethod
    def plot_certainty_count(path, fname, ename, method, mdp, pi_b, baseline_perf, d_size, n_wedge, N, j_max, reps,regret):

        ndsa_hrl = np.load(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_piex_counts.npy')
        ndsa_pib = np.load(f'{path}mdp_data/spihrl_data/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_pib_counts.npy')

        pib_new = []
        hrl_new = []

        for i in range(ndsa_hrl.shape[1]):
            hrl_new.append(np.sum(ndsa_hrl[:,i] > n_wedge, (-1, -2))) 
        for j in range(ndsa_pib.shape[1]):
            pib_new.append(np.sum(ndsa_pib[:,j] > n_wedge, (-1, -2)))


        hrl_new = np.array(hrl_new)
        pib_new = np.array(pib_new)
        plt.figure()
        j = np.maximum(ndsa_hrl.shape[1],ndsa_pib.shape[1])
        plt.plot(np.mean(hrl_new,axis=-1),alpha=.7, label='D=D+pi_ex',marker='o')
        plt.plot(np.mean(pib_new,axis=-1),alpha=.7, label='D=D+pi_b', marker='s')

        plt.legend(loc='lower right')

        plt.title(f'New (s,a) pairs visited N_wedge times')
        plt.ylabel('Number of (s,a) pairs')
        plt.xlabel('Number of Trajectories')
        plt.xticks(np.arange(j,step=10), labels=np.arange(j,step=10)*N+d_size,rotation=45)

        Path(f'{path}mdp_data/figs/{fname}/{ename}/{method}').mkdir(parents=True,exist_ok=True)
        plt.savefig(f'{path}mdp_data/figs/{fname}/{ename}/{method}/{ename}_b{baseline_perf}_D{d_size}_Nw{n_wedge}_N{N}_j{j_max}_r{reps}_reg{regret}_certaincounts.png')

class data:

    # Generate trials
    @staticmethod
    def run_sim(agent, path, fname, ename, num_traj=50, num_reps=5):
        
        data_dict = [[] for i in range(num_reps)]
        for i in range(num_reps):
            print(i)
            ag = copy.deepcopy(agent)
            for j in range(num_traj):
                print(j)
                data_dict[i].append(ag.run())
                
        Path(f'{path}data/{agent.name}/{fname}/{ename}/').mkdir(parents=True,exist_ok=True)
        np.save(f'{path}data/{agent.name}/{fname}/{ename}/trial.npy', data_dict)


    def plot_perf(agent, path, fname, ename):

        data_dict = np.load(f'{path}data/{agent.name}/{fname}/{ename}/trial.npy', allow_pickle=True)
        
        opt_pol, v_opt = utils.value_improvement(agent.mdp)
        v_rand = utils.v_eval(agent.mdp,agent.mdp.rand_pol())

        avg_nsas = []
        avg_nsa = []
        avg_phat = []
        avg_rhat = []
        avg_pistar = []
        avg_perf = []
        for i in range(len(data_dict)):
            nsas = []
            nsa = []
            phat = []
            rhat = []
            pistar = []
            perf = []
            for j in range(len(data_dict[0])):
                nsas.append(data_dict[i][j]['nsas'])
                nsa.append(data_dict[i][j]['nsa'])
                phat.append(data_dict[i][j]['phat'])
                rhat.append(data_dict[i][j]['rhat'])
                pistar.append(data_dict[i][j]['pistar'])

                perf.append(build.compute_performance(agent.mdp, data_dict[i][j]['pistar'], v_opt,v_rand))

            avg_nsas.append(nsas)
            avg_nsa.append(nsa)
            avg_phat.append(phat)
            avg_rhat.append(rhat)
            avg_pistar.append(pistar)
            avg_perf.append(perf)

        plt.figure()
        plt.plot(np.mean(avg_perf,axis=0),alpha=.7, label=f'{agent.name} Exploration',marker='o')
        plt.legend(loc='lower right')

        plt.title(f'Average Expected Return among Exploration methods')
        plt.ylabel('Average Performance')
        plt.xlabel('Number of Trajectories')
        plt.xticks(np.arange(j, step=10), labels=np.arange(j,step=10)*N+d_size,rotation=45)

        plt.show()

    def plot_perf_all(path, agents, fname, ename):
        data_dict = [np.load(f'{path}data/{a.name}/{fname}/{ename}/trial.npy', allow_pickle=True) for a in agents]
        
        opt_pol, v_opt = utils.value_improvement(agents[0].mdp)
        v_rand = utils.v_eval(agents[0].mdp,agents[0].mdp.rand_pol())

        plt.figure()

        for k in range(len(agents)):
            avg_nsas = []
            avg_nsa = []
            avg_phat = []
            avg_rhat = []
            avg_pistar = []
            avg_perf = []
            for i in range(len(data_dict)):
                nsas = []
                nsa = []
                phat = []
                rhat = []
                pistar = []
                perf = []
                for j in range(len(data_dict[k][0])):
                    nsas.append(data_dict[k][i][j]['nsas'])
                    nsa.append(data_dict[k][i][j]['nsa'])
                    phat.append(data_dict[k][i][j]['phat'])
                    rhat.append(data_dict[k][i][j]['rhat'])
                    pistar.append(data_dict[k][i][j]['pistar'])

                    perf.append(build.compute_performance(agents[k].mdp, data_dict[k][i][j]['pistar'], v_opt,v_rand))

                avg_nsas.append(nsas)
                avg_nsa.append(nsa)
                avg_phat.append(phat)
                avg_rhat.append(rhat)
                avg_pistar.append(pistar)
                avg_perf.append(perf)
        
            plt.plot(np.mean(avg_perf,axis=0),alpha=.7, label=f'{agents[k].name} Exploration',marker='o')

        plt.legend(loc='lower right')

        plt.title(f'Average Expected Return among Exploration methods')
        plt.ylabel('Average Performance')
        plt.xlabel('Number of Trajectories')
        plt.xticks(np.arange(j, step=10), labels=np.arange(j,step=10)*N+d_size,rotation=45)

        plt.show()
        Path(f'{path}figs/{fname}/{ename}/').mkdir(parents=True,exist_ok=True)
        plt.savefig(f'{path}figs/{fname}/{ename}/trial.png')



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

grid = args.grid

movement_reward = args.movement_reward
goal_reward = args.goal_reward
obstacle_reward = args.obstacle_reward

#path = '/nfs/hpc/share/soloww/spi_hrl/'
path = args.path

# Load MDP configuration
P = np.load(f'{path}configs/{env_type}/{env}_P.npy')
R = np.load(f'{path}configs/{env_type}/{env}_R.npy')
init_state = np.load(f'{path}configs/{env_type}/{env}_init.npy')
goal_states = np.load(f'{path}configs/{env_type}/{env}_goal.npy')
loaded_mdp = mdp(P, R, init_state, goal_states)

print('loaded_mdp')

pib = build.gen_baseline(loaded_mdp, baseline)
D = build.build_D(loaded_mdp, pib, d_size)

ucbvi_agent = UCBVIAgent(loaded_mdp, D=D, delta=.95)
rmax_agent = RMAXAgent(loaded_mdp, D=D, m=20)
rfnpd_agent = RFNPDAgent(loaded_mdp, D=D, theta=10)
bonusrmax_agent = bonusRMAXAgent(loaded_mdp, D=D, m=20)
viec_agent = VIECAgent(loaded_mdp,m=2)

agents = [rmax_agent, bonusrmax_agent]

for a in agents:
    print(f'{a.name}')
    data.run_sim(a, path, env_type, env, num_traj=250, num_reps=30)

data.plot_perf_all(path, agents, env_type, env)


sys.exit(0)






                            

    
