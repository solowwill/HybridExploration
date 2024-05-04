# Code for plotting data
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

class utils:
    S=0
    A=1
    R=2
    S_P=3   

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
        parser.add_argument("--num_reps", type=int, default=10, help="Number of Reps")
        parser.add_argument("--num_traj", type=int, default=500, help="Number of Trajectories")
        parser.add_argument("--step", type=int, default=25, help="Trajectory Step for Saving")
        parser.add_argument("--baseline_num", type=int, default=0, help="Baseline Num start")
        parser.add_argument("--dataset_num", type=int, default=0, help="Dataset Num start")


        # Path to data
        parser.add_argument("--path", type=str, default="")

        # Parameter for uncertainty
        parser.add_argument("--m", type=int, default=20, help="Uncertainty Paramter")
        

        args = parser.parse_args()
        return args
    
    @staticmethod
    def approx_q_eval(mdp, P_hat, R_hat, policy, epsilon=.01):

        q = np.zeros((P_hat.shape[0],P_hat.shape[1]))
        
        for _ in range(100):

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
            
        return q
    
    # Approximate Q Improvement given an MLE MDP
    @staticmethod 
    def approx_q_improvement(mdp, P_hat, R_hat, epsilon=.01):

        q = np.zeros((P_hat.shape[0],P_hat.shape[1]))
        
        for _ in range(100):
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
        '''policy = np.zeros((mdp.S,mdp.A))
        best_actions = np.argmax(action_reward,axis=-1).flatten()
        for s in range(mdp.S):
            policy[s,best_actions[s]] = 1'''
        return policy, q
    

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
    
    # Compute the SPIBB-optimal policy
    @staticmethod 
    def spibb(mdp, p_hat, r_hat, nsa, pib, m, epsilon=.01):
        q = np.zeros((mdp.S, mdp.A))
        for _ in range(100):
            old_q = q

            spibb_pol = utils.greedy_q_projection(q, nsa, pib, m)
            
            q = utils.approx_q_eval(mdp, p_hat, r_hat, spibb_pol)

            if np.max(np.abs(old_q-q)) < epsilon:
                break

        return spibb_pol, q


    # Perform greedy q projection for SPIBB
    @staticmethod
    def greedy_q_projection(q, n_sa, pib, m):

        spibb_pol = np.zeros(pib.shape)

        for s in range(pib.shape[0]):
            for a in range(pib.shape[1]):

                # Bootstrap with baseline policy in uncertain states
                if n_sa[s,a] < m:
                        spibb_pol[s,a] = pib[s,a]

            # Perform the greedy Q projection onto approx MDP
            safe_a = np.argwhere(n_sa[s,:] >= m).flatten()
            if len(safe_a) != 0:
                spibb_pol[s,safe_a[np.argmax(q[s,safe_a])]] = np.sum(pib[s,safe_a])
                
        return spibb_pol
    

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
    
    # Compute the normalized performance from approx dynamics
    @staticmethod
    def performance(mdp, nsas, r_hat, v_opt, v_rand, pib=None, m=None, spibb_flag=False):
        nsa = np.sum(nsas,axis=-1)
        p_hat = nsas / np.where(nsa > 0, nsa, 1)[:,:,np.newaxis]

        if spibb_flag:
            pol, q = utils.spibb(mdp, p_hat, r_hat, nsa, pib, m)
        else:
            pol, q = utils.approx_q_improvement(mdp, p_hat, r_hat)

        return utils.compute_performance(mdp, pol, v_opt, v_rand)

    @staticmethod
    def update_counts(n_sas, trajectory):
        for sarcs in trajectory:
            n_sas[int(sarcs[utils.S]), int(sarcs[utils.A]), int(sarcs[utils.S_P])] += 1
        return n_sas

# Main MDP class 
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

def load_data(path, env_type, env, num_baselines, num_d, num_reps, d_size, baseline, m,num_traj,step, b_num, d_num):
    try:
        perf_avg = np.load(f'{path}figures/{env_type}/{env}/b{baseline}_d{d_size}_m{m}_nb{num_baselines}_nd{num_d}_nr{num_reps}_bn{b_num}_dn{d_num}_perf_avg.npy')
        perf_std = np.load(f'{path}figures/{env_type}/{env}/b{baseline}_d{d_size}_m{m}_nb{num_baselines}_nd{num_d}_nr{num_reps}_bn{b_num}_dn{d_num}_perf_std.npy')
        spibb_avg = np.load(f'{path}figures/{env_type}/{env}/b{baseline}_d{d_size}_m{m}_nb{num_baselines}_nd{num_d}_nr{num_reps}_bn{b_num}_dn{d_num}_spibb_avg.npy')
        spibb_std = np.load(f'{path}figures/{env_type}/{env}/b{baseline}_d{d_size}_m{m}_nb{num_baselines}_nd{num_d}_nr{num_reps}_bn{b_num}_dn{d_num}_spibb_std.npy')
    except:
        # Load MDP configuration
        P = np.load(f'{path}configs/{env_type}/{env}_P.npy')
        R = np.load(f'{path}configs/{env_type}/{env}_R.npy')
        init_state = np.load(f'{path}configs/{env_type}/{env}_init.npy')
        goal_states = np.load(f'{path}configs/{env_type}/{env}_goal.npy')
        loaded_mdp = mdp(P, R, init_state, goal_states)

        _, v_opt = utils.value_improvement(loaded_mdp)
        v_rand = utils.v_eval(loaded_mdp,loaded_mdp.rand_pol())

        perf = np.empty(shape=(num_baselines, num_d, num_reps, int(num_traj/step), 6))
        perf_spibb = np.empty(shape=(num_baselines, num_d, num_reps, int(num_traj/step), 6))
        for b in range(b_num,b_num+num_baselines):
            for d in range(d_num,d_num+num_d):
                for r in range(num_reps):
                    print(f'b{b}, d{d}, r{r}')
                    pib = np.load(f'{path}config_data/{env_type}/{env}/b{baseline}/b{b}.npy')

                    rmax_nsas = np.load(f'{path}data/{env_type}/{env}/b{baseline}/b{b}/ds{d_size}/d{d}/rmax/r{r}_nsas.npy')
                    rmax_rhat = np.load(f'{path}data/{env_type}/{env}/b{baseline}/b{b}/ds{d_size}/d{d}/rmax/r{r}_rhat.npy')

                    bonusrmax_nsas = np.load(f'{path}data/{env_type}/{env}/b{baseline}/b{b}/ds{d_size}/d{d}/bonusrmax/r{r}_nsas.npy')
                    bonusrmax_rhat = np.load(f'{path}data/{env_type}/{env}/b{baseline}/b{b}/ds{d_size}/d{d}/bonusrmax/r{r}_rhat.npy')
                        
                    pib_nsas = np.load(f'{path}data/{env_type}/{env}/b{baseline}/b{b}/ds{d_size}/d{d}/pib/r{r}_nsas.npy')
                    pib_rhat = np.load(f'{path}data/{env_type}/{env}/b{baseline}/b{b}/ds{d_size}/d{d}/pib/r{r}_rhat.npy')

                    spibb_nsas = np.load(f'{path}data/{env_type}/{env}/b{baseline}/b{b}/ds{d_size}/d{d}/spibb/r{r}_nsas.npy')
                    spibb_rhat = np.load(f'{path}data/{env_type}/{env}/b{baseline}/b{b}/ds{d_size}/d{d}/spibb/r{r}_rhat.npy')

                    fqi_nsas = np.load(f'{path}data/{env_type}/{env}/b{baseline}/b{b}/ds{d_size}/d{d}/fqi/r{r}_nsas.npy')
                    fqi_rhat = np.load(f'{path}data/{env_type}/{env}/b{baseline}/b{b}/ds{d_size}/d{d}/fqi/r{r}_rhat.npy')
                        
                    bonusfqirmax_nsas = np.load(f'{path}data/{env_type}/{env}/b{baseline}/b{b}/ds{d_size}/d{d}/bonusfqirmax/r{r}_nsas.npy')
                    bonusfqirmax_rhat = np.load(f'{path}data/{env_type}/{env}/b{baseline}/b{b}/ds{d_size}/d{d}/bonusfqirmax/r{r}_rhat.npy')

                    rmax_perf = [utils.performance(loaded_mdp, rmax_nsas[i], rmax_rhat[i], v_opt, v_rand) for i in range(int(num_traj/step))]
                    bonusrmax_perf = [utils.performance(loaded_mdp, bonusrmax_nsas[i], bonusrmax_rhat[i], v_opt, v_rand) for i in range(int(num_traj/step))]
                    pib_perf = [utils.performance(loaded_mdp, pib_nsas[i], pib_rhat[i], v_opt, v_rand) for i in range(int(num_traj/step))]
                    spibb_perf = [utils.performance(loaded_mdp, spibb_nsas[i], spibb_rhat[i], v_opt, v_rand) for i in range(int(num_traj/step))]
                    fqi_perf = [utils.performance(loaded_mdp, fqi_nsas[i], fqi_rhat[i], v_opt, v_rand) for i in range(int(num_traj/step))]
                    bonusfqirmax_perf = [utils.performance(loaded_mdp, bonusfqirmax_nsas[i], bonusfqirmax_rhat[i], v_opt, v_rand) for i in range(int(num_traj/step))]

                    perf[b,d,r,:,:] = np.array([rmax_perf, bonusrmax_perf, pib_perf, spibb_perf, fqi_perf, bonusfqirmax_perf]).T

                    rmax_perf = [utils.performance(loaded_mdp, rmax_nsas[i], rmax_rhat[i], v_opt, v_rand) for i in range(int(num_traj/step))]
                    bonusrmax_perf = [utils.performance(loaded_mdp, bonusrmax_nsas[i], bonusrmax_rhat[i], v_opt, v_rand) for i in range(int(num_traj/step))]
                    pib_perf = [utils.performance(loaded_mdp, pib_nsas[i], pib_rhat[i], v_opt, v_rand) for i in range(int(num_traj/step))]
                    spibb_perf = [utils.performance(loaded_mdp, spibb_nsas[i], spibb_rhat[i], v_opt, v_rand) for i in range(int(num_traj/step))]
                    fqi_perf = [utils.performance(loaded_mdp, fqi_nsas[i], fqi_rhat[i], v_opt, v_rand) for i in range(int(num_traj/step))]
                    bonusfqirmax_perf = [utils.performance(loaded_mdp, bonusfqirmax_nsas[i], bonusfqirmax_rhat[i], v_opt, v_rand) for i in range(int(num_traj/step))]

                    rmax_spibb = [utils.performance(loaded_mdp, rmax_nsas[i], rmax_rhat[i], v_opt, v_rand, pib=pib, m=m, spibb_flag=True) for i in range(int(num_traj/step))]
                    bonusrmax_spibb = [utils.performance(loaded_mdp, bonusrmax_nsas[i], bonusrmax_rhat[i], v_opt, v_rand, pib=pib, m=m, spibb_flag=True) for i in range(int(num_traj/step))]
                    pib_spibb = [utils.performance(loaded_mdp, pib_nsas[i], pib_rhat[i], v_opt, v_rand, pib=pib, m=m, spibb_flag=True) for i in range(int(num_traj/step))]
                    spibb_spibb = [utils.performance(loaded_mdp, spibb_nsas[i], spibb_rhat[i], v_opt, v_rand, pib=pib, m=m, spibb_flag=True) for i in range(int(num_traj/step))]
                    fqi_spibb = [utils.performance(loaded_mdp, fqi_nsas[i], fqi_rhat[i], v_opt, v_rand, pib=pib, m=m, spibb_flag=True) for i in range(int(num_traj/step))]
                    bonusfqirmax_spibb = [utils.performance(loaded_mdp, bonusfqirmax_nsas[i], bonusfqirmax_rhat[i], v_opt, v_rand, pib=pib, m=m, spibb_flag=True) for i in range(int(num_traj/step))]

                    perf_spibb[b,d,r,:,:] = np.array([rmax_spibb, bonusrmax_spibb, pib_spibb, spibb_spibb, fqi_spibb, bonusfqirmax_spibb]).T

        perf_avg = np.mean(perf, axis=(0,1,2))
        perf_std = np.std(perf, axis=(0,1,2))

        spibb_avg = np.mean(perf_spibb, axis=(0,1,2))
        spibb_std = np.std(perf_spibb, axis=(0,1,2))

        Path(f'{path}figures/{env_type}/{env}/').mkdir(parents=True,exist_ok=True)
        np.save(f'{path}figures/{env_type}/{env}/b{baseline}_d{d_size}_m{m}_nb{num_baselines}_nd{num_d}_nr{num_reps}_bn{b_num}_dn{d_num}_perf_avg.npy',perf_avg)
        np.save(f'{path}figures/{env_type}/{env}/b{baseline}_d{d_size}_m{m}_nb{num_baselines}_nd{num_d}_nr{num_reps}_bn{b_num}_dn{d_num}_perf_std.npy',perf_std)
        np.save(f'{path}figures/{env_type}/{env}/b{baseline}_d{d_size}_m{m}_nb{num_baselines}_nd{num_d}_nr{num_reps}_bn{b_num}_dn{d_num}_spibb_avg.npy', spibb_avg)
        np.save(f'{path}figures/{env_type}/{env}/b{baseline}_d{d_size}_m{m}_nb{num_baselines}_nd{num_d}_nr{num_reps}_bn{b_num}_dn{d_num}_spibb_std.npy',spibb_std)

    return perf_avg, perf_std, spibb_avg, spibb_std

def main():
    args = utils.parse_args()

    env_type = args.env_type
    env = args.env

    baseline = args.baseline
    d_size = args.d_size

    num_baselines = args.num_baselines
    num_d = args.num_datasets
    num_reps = args.num_reps
    num_traj = args.num_traj
    step = args.step
    b_num = args.baseline_num
    d_num = args.dataset_num

    m = args.m

    path = args.path

    perf_avg, perf_std, spibb_avg, spibb_std = load_data(path, env_type, env, num_baselines, num_d, num_reps, d_size, baseline, m, num_traj, step, b_num, d_num)

    P = np.load(f'{path}configs/{env_type}/{env}_P.npy')
    R = np.load(f'{path}configs/{env_type}/{env}_R.npy')
    init_state = np.load(f'{path}configs/{env_type}/{env}_init.npy')
    goal_states = np.load(f'{path}configs/{env_type}/{env}_goal.npy')
    loaded_mdp = mdp(P, R, init_state, goal_states)
    pib = np.load(f'{path}config_data/{env_type}/{env}/b{baseline}/b{1}.npy')
    D = np.load(f'{path}config_data/{env_type}/{env}/b{baseline}/b{1}/ds{d_size}/d{1}.npy',allow_pickle=True)
    _, v_opt = utils.value_improvement(loaded_mdp)
    v_rand = utils.v_eval(loaded_mdp,loaded_mdp.rand_pol())
    
    nsas = np.zeros((loaded_mdp.S, loaded_mdp.A, loaded_mdp.S))
    for trajectory in D:
        nsas = utils.update_counts(nsas, trajectory)  
    base_perf = utils.performance(loaded_mdp, nsas, loaded_mdp.R, v_opt, v_rand)   
    base_spibb = utils.performance(loaded_mdp, nsas, loaded_mdp.R, v_opt, v_rand, pib=pib, m=m, spibb_flag=True)

    perf_avg = np.concatenate((np.tile(base_perf,6)[np.newaxis,:], perf_avg),axis=0)
    perf_std = np.concatenate((np.tile(0,6)[np.newaxis,:], perf_std),axis=0)
    spibb_avg = np.concatenate((np.tile(base_spibb,6)[np.newaxis,:], spibb_avg),axis=0)
    spibb_std = np.concatenate((np.tile(0,6)[np.newaxis,:], spibb_std),axis=0)

    fig,ax = plt.subplots(figsize=(8,6))
    x = np.arange(start=d_size, stop=num_traj+d_size+step, step=step)
    ax.plot(x, perf_avg[:,0],label='RMAX (Online Ex)')
    ax.plot(x, perf_avg[:,1], label='Bonus-RMAX (Hybrid Ex)')
    ax.plot(x, perf_avg[:,2],label='Sample with PiB')
    ax.plot(x, perf_avg[:,3], label='Sample w SPIBB Pol')
    ax.plot(x, perf_avg[:,4], label='HyQ:FQI (Hybrid Ex)')
    ax.plot(x, perf_avg[:,5], label='Factored-Bonus-RMAX (Hybrid Ex)')

    ax.fill_between(x, perf_avg[:,0]-perf_std[:,0], perf_avg[:,0]+perf_std[:,0],alpha=.5)
    ax.fill_between(x, perf_avg[:,1]-perf_std[:,1], perf_avg[:,1]+perf_std[:,1],alpha=.5)
    ax.fill_between(x, perf_avg[:,2]-perf_std[:,2], perf_avg[:,2]+perf_std[:,2],alpha=.5)
    ax.fill_between(x, perf_avg[:,3]-perf_std[:,3], perf_avg[:,3]+perf_std[:,3],alpha=.5)
    ax.fill_between(x, perf_avg[:,4]-perf_std[:,4], perf_avg[:,4]+perf_std[:,4],alpha=.5)
    ax.fill_between(x, perf_avg[:,5]-perf_std[:,5], perf_avg[:,5]+perf_std[:,5],alpha=.5)

    ax.legend(loc='lower right')
    ax.set_xlabel('Number of Trajectories')
    ax.set_ylabel('Average Normalized Performance')
    ax.set_xticks(x[::5], labels=x[::5],rotation=45)
    ax.set_title('Average Performance with Different Exploration Schemes')

    plt.savefig(f'{path}figures/{env_type}/{env}/b{baseline}_d{d_size}_m{m}_b{num_baselines}_d{num_d}_r{num_reps}_bn{b_num}_dn{d_num}_perf.png')
    plt.close()

    fig,ax = plt.subplots(figsize=(8,6))
    x = np.arange(start=d_size, stop=num_traj+d_size+step, step=step)
    ax.plot(x, spibb_avg[:,0],label='RMAX (Online Ex)')
    ax.plot(x, spibb_avg[:,1], label='Bonus-RMAX (Hybrid Ex)')
    ax.plot(x, spibb_avg[:,2],label='Sample with PiB')
    ax.plot(x, spibb_avg[:,3], label='Sample w SPIBB Pol')
    ax.plot(x, spibb_avg[:,4], label='HyQ:FQI (Hybrid Ex)')
    ax.plot(x, spibb_avg[:,5], label='Factored-Bonus-RMAX (Hybrid Ex)')

    ax.fill_between(x, spibb_avg[:,0]-spibb_std[:,0], spibb_avg[:,0]+spibb_std[:,0],alpha=.5)
    ax.fill_between(x, spibb_avg[:,1]-spibb_std[:,1], spibb_avg[:,1]+spibb_std[:,1],alpha=.5)
    ax.fill_between(x, spibb_avg[:,2]-spibb_std[:,2], spibb_avg[:,2]+spibb_std[:,2],alpha=.5)
    ax.fill_between(x, spibb_avg[:,3]-spibb_std[:,3], spibb_avg[:,3]+spibb_std[:,3],alpha=.5)
    ax.fill_between(x, spibb_avg[:,4]-spibb_std[:,4], spibb_avg[:,4]+spibb_std[:,4],alpha=.5)
    ax.fill_between(x, spibb_avg[:,5]-spibb_std[:,5], spibb_avg[:,5]+spibb_std[:,5],alpha=.5)

    ax.legend(loc='lower right')
    ax.set_xlabel('Number of Trajectories')
    ax.set_ylabel('Average Normalized SPIBB Policy Performance')
    ax.set_xticks(x[::5], labels=x[::5],rotation=45)
    ax.set_title('Average SPIBB Poliycy Performance with Different Exploration Schemes')

    plt.savefig(f'{path}figures/{env_type}/{env}/b{baseline}_d{d_size}_m{m}_b{num_baselines}_d{num_d}_r{num_reps}_bn{b_num}_dn{d_num}_spibb.png')
    plt.close()
    

if __name__ == '__main__':
    main()