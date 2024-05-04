# Main script for running experiments
import numpy as np
import argparse
from pathlib import Path
import agents
import multiprocessing as mp

class utils:
    @staticmethod
    def parse_args():
        
        parser = argparse.ArgumentParser()

        parser.add_argument("--i", type=int, default="0", help="id")
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


        # Path to data
        parser.add_argument("--path", type=str, default="")

        # Parameter for uncertainty
        parser.add_argument("--m", type=int, default=20, help="Uncertainty Paramter")
        
        args = parser.parse_args()
        return args

# Main MDP class 
class mdp:

    # Set indices of trajectory tuples 
    S = 0
    A = 1
    R = 2
    S_P = 3

    def __init__(self, P, R, init_state, goal_states, state_space=None, action_space=None, gamma=.95, H=100):

        self.goal_states = goal_states
        self.init_state = init_state

        self.state_space = state_space
        self.action_space = action_space

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

def run(id, num_d, num_reps, path, env_type, env, baseline, d_size, loaded_mdp, m, num_traj, step):
    pib = np.load(f'{path}config_data/{env_type}/{env}/b{baseline}/b{id}.npy')
    print(f'running {id}')
    # For each dataset
    for j in range(num_d):

        D = np.load(f'{path}config_data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}.npy',allow_pickle=True)

        # For each repetition
        for i in range(num_reps):
            print(f'id{i}, d{j}, r{i}')
            rmax_agent = agents.RMAXAgent(loaded_mdp, D=D, m=m, pib=pib)
            bonusrmax_agent = agents.bonusRMAXAgent(loaded_mdp, D=D, m=m, pib=pib)
            pib_agent = agents.spiPiBAgent(loaded_mdp, D=D, m=m, pib=pib)
            spibb_agent = agents.spiSPIBBAgent(loaded_mdp, D=D, m=m, pib=pib)
            fqi_agent = agents.FQIAgent(loaded_mdp, D=D, m=m, pib=pib)
            bonusfqirmax_agent = agents.bonusFQIRMAXAgent(loaded_mdp, D=D, m=m, pib=pib)

            Path(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/rmax/').mkdir(parents=True,exist_ok=True)
            Path(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/bonusrmax/').mkdir(parents=True,exist_ok=True)
            Path(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/pib/').mkdir(parents=True,exist_ok=True)
            Path(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/spibb/').mkdir(parents=True,exist_ok=True)
            Path(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/fqi/').mkdir(parents=True,exist_ok=True)
            Path(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/bonusfqirmax/').mkdir(parents=True,exist_ok=True)

            nsas_shape = (1,rmax_agent.P_hat.shape[0], rmax_agent.P_hat.shape[1],rmax_agent.P_hat.shape[2])
            rhat_shape = (1, rmax_agent.R_hat.shape[0], rmax_agent.R_hat.shape[1])
            rmax_nsas = np.empty(shape=nsas_shape)
            rmax_rhat = np.empty(shape=rhat_shape)
            bonusrmax_nsas = np.empty(shape=nsas_shape)
            bonusrmax_rhat = np.empty(shape=rhat_shape)
            pib_nsas = np.empty(shape=nsas_shape)
            pib_rhat = np.empty(shape=rhat_shape)
            spibb_nsas = np.empty(shape=nsas_shape)
            spibb_rhat = np.empty(shape=rhat_shape)
            fqi_nsas = np.empty(shape=nsas_shape)
            fqi_rhat = np.empty(shape=rhat_shape)
            bonusfqirmax_nsas = np.empty(shape=nsas_shape)
            bonusfqirmax_rhat = np.empty(shape=rhat_shape)

            for k in range(num_traj):
                rmax_dict = rmax_agent.run()
                bonusrmax_dict = bonusrmax_agent.run()
                pib_dict = pib_agent.run()
                spibb_dict = spibb_agent.run()
                fqi_dict = fqi_agent.run()
                bonusfqirmax_dict = bonusfqirmax_agent.run()
                
                if k % step == 0:

                    rmax_nsas = np.concatenate((rmax_nsas, rmax_dict['nsas'][np.newaxis,:]),axis=0)
                    rmax_rhat = np.concatenate((rmax_rhat, rmax_dict['rhat'][np.newaxis,:]),axis=0)
                    bonusrmax_nsas = np.concatenate((bonusrmax_nsas, bonusrmax_dict['nsas'][np.newaxis,:]),axis=0)
                    bonusrmax_rhat = np.concatenate((bonusrmax_rhat, bonusrmax_dict['rhat'][np.newaxis,:]),axis=0)
                    pib_nsas = np.concatenate((pib_nsas, pib_dict['nsas'][np.newaxis,:]),axis=0)
                    pib_rhat = np.concatenate((pib_rhat, pib_dict['rhat'][np.newaxis,:]),axis=0)
                    spibb_nsas = np.concatenate((spibb_nsas, spibb_dict['nsas'][np.newaxis,:]),axis=0)
                    spibb_rhat = np.concatenate((spibb_rhat, spibb_dict['rhat'][np.newaxis,:]),axis=0)
                    fqi_nsas = np.concatenate((fqi_nsas, fqi_dict['nsas'][np.newaxis,:]),axis=0)
                    fqi_rhat = np.concatenate((fqi_rhat, fqi_dict['rhat'][np.newaxis,:]),axis=0)
                    bonusfqirmax_nsas = np.concatenate((bonusfqirmax_nsas, bonusfqirmax_dict['nsas'][np.newaxis,:]),axis=0)
                    bonusfqirmax_rhat = np.concatenate((bonusfqirmax_rhat, bonusfqirmax_dict['rhat'][np.newaxis,:]),axis=0)

            np.save(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/rmax/r{i}_nsas.npy', rmax_nsas.astype('uint16'))
            np.save(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/bonusrmax/r{i}_nsas.npy', bonusrmax_nsas.astype('uint16'))
            np.save(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/pib/r{i}_nsas.npy', pib_nsas.astype('uint16'))
            np.save(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/spibb/r{i}_nsas.npy', spibb_nsas.astype('uint16'))
            np.save(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/fqi/r{i}_nsas.npy', fqi_nsas.astype('uint16'))
            np.save(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/bonusfqirmax/r{i}_nsas.npy', bonusfqirmax_nsas.astype('uint16'))

            np.save(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/rmax/r{i}_rhat.npy', rmax_rhat.astype('float32'))
            np.save(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/bonusrmax/r{i}_rhat.npy', bonusrmax_rhat.astype('float32'))
            np.save(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/pib/r{i}_rhat.npy', pib_rhat.astype('float32'))
            np.save(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/spibb/r{i}_rhat.npy', spibb_rhat.astype('float32'))
            np.save(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/fqi/r{i}_rhat.npy', fqi_rhat.astype('float32'))
            np.save(f'{path}data/{env_type}/{env}/b{baseline}/b{id}/ds{d_size}/d{j}/bonusfqirmax/r{i}_rhat.npy', bonusfqirmax_rhat.astype('float32'))

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

    m = args.m


    path = args.path
    #path = '/nfs/hpc/share/soloww/spi_hpc/'

    # Load MDP configuration
    P = np.load(f'{path}configs/{env_type}/{env}_P.npy')
    R = np.load(f'{path}configs/{env_type}/{env}_R.npy')
    init_state = np.load(f'{path}configs/{env_type}/{env}_init.npy')
    goal_states = np.load(f'{path}configs/{env_type}/{env}_goal.npy')
    state_space = np.load(f'{path}configs/{env_type}/{env}_statespace.npy')
    action_space = np.load(f'{path}configs/{env_type}/{env}_actionspace.npy')
    loaded_mdp = mdp(P, R, init_state, goal_states, state_space=state_space, action_space=action_space)

    processes = [mp.Process(target=run, args=(i, num_d, num_reps, path, env_type, env, baseline, d_size, loaded_mdp, m, num_traj, step)) for i in range(num_baselines)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

if __name__ == '__main__':
    main()