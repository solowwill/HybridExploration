# Main file for data generation
import numpy as np
import argparse
from pathlib import Path

# General class for an MDP 
class mdp:

    # Set indices of trajectory tuples 
    S = 0
    A = 1
    R = 2
    S_P = 3

    def __init__(self, P, R, init_state, goal_states, gamma=.95, H=100):

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
    def gen_baseline(mdp, targ_performance, dec=.05):

        opt_pol, q = utils.q_improvement(mdp)
        v_opt = np.sum(opt_pol * q, axis=-1)
        v_rand = utils.v_eval(mdp,mdp.rand_pol())

        pi_b = np.copy(opt_pol)
        pib_perf = build.compute_performance(mdp,pi_b, v_opt, v_rand)

        # Iteratively remove action weight from the best action
        while pib_perf > targ_performance:

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
                else:
                    break

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

        # Dataset Params
        parser.add_argument("--baseline", type=float, default=.5, help="Baseline Performance")
        parser.add_argument("--d_size", type=int, default=10, help="Init Dataset Size")
        
        # Experiement params
        parser.add_argument("--num_baselines", type=int, default=10, help="Number of Baselines")
        parser.add_argument("--num_datasets", type=int, default=10, help="Number of Datasets")

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


        return np.maximum(spibb.V[s] - spibb.Q[s,a],0)
    
def main():
    args = utils.parse_args()

    env_type = args.env_type
    env = args.env

    baseline = args.baseline
    d_size = args.d_size

    num_baselines = args.num_baselines
    num_d = args.num_datasets

    path = args.path
    #path = '/nfs/hpc/share/soloww/spi_hpc/'

    # Load MDP configuration
    P = np.load(f'{path}configs/{env_type}/{env}_P.npy')
    R = np.load(f'{path}configs/{env_type}/{env}_R.npy')
    init_state = np.load(f'{path}configs/{env_type}/{env}_init.npy')
    goal_states = np.load(f'{path}configs/{env_type}/{env}_goal.npy')
    loaded_mdp = mdp(P, R, init_state, goal_states)

    for i in range(num_baselines):
        try:
            b = np.load(f'{path}config_data/{env_type}/{env}/b{baseline}/b{i}.npy')
        except:
            b = build.gen_baseline(loaded_mdp, baseline)
            Path(f'{path}config_data/{env_type}/{env}/b{baseline}/').mkdir(parents=True,exist_ok=True)
            np.save(f'{path}config_data/{env_type}/{env}/b{baseline}/b{i}.npy', b)
        
        print(f'Baseline {i}')
        Path(f'{path}config_data/{env_type}/{env}/b{baseline}/b{i}/ds{d_size}/').mkdir(parents=True,exist_ok=True)

        for j in range(num_d):
            d = build.build_D(loaded_mdp, b, d_size)
            np.save(f'{path}config_data/{env_type}/{env}/b{baseline}/b{i}/ds{d_size}/d{j}.npy', np.array(d,dtype=object))

if __name__ == '__main__':
    main()
