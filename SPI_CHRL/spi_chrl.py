# This file allows for the Safe Policy Improvement with Baseline Bootstrapping 

import regret
import spibb
import active_exploration as ex
import mdp  as MDP
import numpy as np
import spibb
import active_exploration
import sys


# Initialize the spi_chrl class to run the algorithm 
class spi_chrl():

    def __init__(self, mdp, pi_b, D, C, N, N_wedge, epsilon):

        self.mdp = mdp
        self.N = N
        self.epsilon = epsilon

        # Create the learner agent that stores what it learns
        self.learner = learner(mdp, pi_b, D, C)

        # Create the object to perform spibb
        self.c_spibb = spibb.constrained_spibb(N_wedge, pi_b)

        # Create the object that computes the exploration policy 
        self.explore = ex.DFE(self.mdp)

        # Create the object that performs c_hat optimization 
        self.mcr = regret.MCR(mdp)

        return 
    
    
    def run(self):
        j = 0
        while j < 2:
            # Infer c_hat by solving the MCR problem
            self.learner.c_hat =  self.mcr.compute_C_hat(self.learner.D, self.learner.C_hat)

            print(self.learner.C_hat)


            # Improve the policy using C-SPIBB
            self.c_spibb.run_cspibb(self.learner.P_hat, self.learner.R_hat, self.mdp.gamma, self.learner.Nsa_D, self.learner.D, self.learner.C_hat)

            for i in range(self.N):
                # Compute exploration policy pi_MCR
                pi_mcr = self.explore.pi_mcr(self.learner.C_hat, self.learner.D, self.learner.Nsa_D, self.c_spibb.N_wedge)
                print(pi_mcr)

                # Sample a trajectory following pi_MCR
                trajectory = self.mdp.sample_trajectory(pi_mcr)

                # Add constraint violating pairs to C_hat
                self.learner.update_C_hat(trajectory)

                # Add safe pairs to D
                self.learner.update_D(trajectory)

                # Update P_hat and R_hat
                self.learner.update_counts(trajectory)


                # If the regret for uknown states is less than epsilon
            j+=1

            



# Initialize the learner agent class to store the relevant data
class learner():

    def __init__(self, mdp, pi_b=None, D=[], C=[]):

        self.mdp = mdp

        self.pi_b = pi_b

        self.c_hat = 0

        # Get only the unique elements in C_hat and D
        if len(C) != 0:
            self.C_hat = np.unique(C,axis=0)
        else:
            self.C_hat = np.ndarray(shape=(0,5))
        self.D = np.unique(D,axis=0)


        # Build state counts, dynamics and reward approximation
        self._build_counts()
    
    
    # Add safe pairs to set if not already in the set
    def update_D(self, trajectory):
        
        # Go through every tuple in the trajectory
        for sarcs in trajectory:
            # If not constraint violating, add to D
            if sarcs[MDP.C] <= 0 and not np.all(sarcs != self.D, axis=1).any():
                self.D = np.concatenate((self.D,sarcs[np.newaxis,:]),axis=0)
    

    # Add unsafe pairs if not already in the set 
    def update_C_hat(self, trajectory):
        # Go through every tuple in the trajectory
        for sarcs in trajectory:

            # If  constraint violating, add to C
            if sarcs[MDP.C] > 0 and not np.all(sarcs != self.C_hat,axis=1).any():
                self.C_hat = np.concatenate((self.C_hat,sarcs[np.newaxis,:]),axis=0)


    # Update the state acount pair counts based on the trajectory 
    def update_counts(self, trajectory):
        for sarcs in trajectory:
            self.Nsas_D[int(sarcs[MDP.S]), int(sarcs[MDP.A]), int(sarcs[MDP.S_P])] += 1
            self.Nsa_D[int(sarcs[MDP.S]), int(sarcs[MDP.A])] +=1

            # Update the average of R_hat
            self.R_hat[int(sarcs[MDP.S]),int(sarcs[MDP.A])] += -(self.R_hat[int(sarcs[MDP.S]), int(sarcs[MDP.A])]\
                                                     / self.Nsa_D[int(sarcs[MDP.S]), int(sarcs[MDP.A])]) + \
                                                      (sarcs[MDP.R] / self.Nsa_D[int(sarcs[MDP.S]), int(sarcs[MDP.A])])
            
        # Update P_hat
        self.P_hat = self.Nsas_D / np.where(self.Nsa_D > 0, self.Nsa_D, 1)[:,:,np.newaxis]
    


    # Build state counts, dynamics, reward approximation
    def _build_counts(self):
        self.Nsas_D = np.zeros((self.mdp.S,self.mdp.A,self.mdp.S))
        self.Nsa_D = np.zeros((self.mdp.S,self.mdp.A))
        self.R_hat = np.zeros((self.mdp.S,self.mdp.A))

        # Count each pair 
        for sarcs in self.D:
            self.Nsas_D[int(sarcs[MDP.S]), int(sarcs[MDP.A]), int(sarcs[MDP.S_P])] += 1
            self.Nsa_D[int(sarcs[MDP.S]), int(sarcs[MDP.A])] +=1
            self.R_hat[int(sarcs[MDP.S]), int(sarcs[MDP.A])] += sarcs[MDP.R]
        
        # Average to get P_hat and R_hat
        self.P_hat = self.Nsas_D / np.where(self.Nsa_D > 0, self.Nsa_D, 1)[:,:,np.newaxis]
        self.R_hat = self.R_hat / np.where(self.Nsa_D > 0, self.Nsa_D, 1)

        
