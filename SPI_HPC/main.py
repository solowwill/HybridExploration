import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import sys
import copy
import agents

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
    

    # Explore using the optimal policy on the MLE MDP
    def pi_value_improve(self, N_D, N_wedge, P_hat, R_hat):

        pi_val, q = utils.approx_q_improvement(self.mdp, P_hat, R_hat)  

        # Define approximate policy on unseen states
        unseen_states = np.sum(N_D ,axis=-1) == 0
        pi_val[unseen_states,:] = self.pi_b[unseen_states,:]    
        return pi_val 
    

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

fqi_agent = agents.FQIAgent(loaded_mdp, D=D, m=20, pib=pib)



print(f'{fqi_agent.name}')
data.run_sim(fqi_agent, path, env_type, env, num_traj=250, num_reps=1)

data.plot_perf_all(path, [fqi_agent], env_type, env)


sys.exit(0)






                            

    
