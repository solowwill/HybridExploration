# This file allows for Safe Policy Improvement with Baseline Bootstrapping
import numpy as np
import mdp as MDP

class spibb:

    def __init__(self, N_wedge, N_D, pi_b):
    
        self.N_wedge = N_wedge
        self.N_D  = N_D
        self.pi_b = pi_b

        self.spibb = np.copy(pi_b)

    # Run SPIBB
    def run_spibb(self, P_hat, R_hat, gamma, epsilon=.01):
        self.Q = q_approx(P_hat, R_hat, gamma)

        # Perform until a threshold is met 
        while True:
            # Store the old Q value 
            old_Q = self.Q

            # Compute the new SPIBB policy based on the greedy Q projection
            # of pi_b onto M_hat
            self.spibb = self.greedy_q_projection(self.Q)

            # Recompute the q-function in M_hat with spibb policy
            self.Q = q_approx(P_hat, R_hat, gamma)

            # Exit the loop once sufficiently close approximation
            if np.max(np.abs(old_Q-self.Q)) < epsilon:
                break

    def greedy_q_projection(self, Q):
        # Initialize spibb policy to all zeros
        spibb_pol = np.zeros((self.pi_b.shape[0],self.pi_b.shape[1]))

        # Go through all states and actions
        for s in range(self.pi_b.shape[0]):
            for a in range(self.pi_b.shape[1]):

                # If we do not have sufficient counts, bootstrap with baseline
                if self.N_D[s,a] < self.N_wedge:
                    spibb_pol[s,a] = self.pi_b[s,a]

                # Perform the greedy Q projection onto approx MDP
                else:
                    # Compute the safe actions from the dataset
                    safe_a = np.argwhere(self.N_D[s,:] >= self.N_wedge).flatten()

                    # Get the best action from the safe actions based on Q
                    # And assign that weight to all the other best actions
                    spibb_pol[s,np.argmax(Q[s,safe_a])] = np.sum(self.pi_b[s,safe_a])

        return spibb_pol

    
class constrained_spibb:

    def __init__(self, N_wedge, pi_b):
        self.N_wedge = N_wedge
        self.pi_b = pi_b

        self.spibb = np.copy(pi_b)

    # Run SPIBB
    def run_cspibb(self, P_hat, R_hat, gamma, N_D, D, C, epsilon=.01):

        self.N_D = N_D

        # Set C_hat and D by cutting out everything but state action pairs
        self.D = D[:,[MDP.S,MDP.A]]
        if len(C) == 0:
            self.C_hat = np.ndarray(shape=(0,2))
        else:
            self.C_hat = C[:,[MDP.S,MDP.A]]

        # Approximate the Q function
        self.Q = q_approx(P_hat, R_hat, gamma, self.spibb)

        # Perform until a threshold is met 
        while True:
            # Store the old Q value 
            old_Q = self.Q

            # Compute the new SPIBB policy based on the greedy Q projection
            # of pi_b onto M_hat
            self.spibb = self.c_greedy_q_projection()

            # Recompute the q-function in M_hat with spibb policy
            self.Q = q_approx(P_hat, R_hat, gamma, self.spibb)

            # Exit the loop once sufficiently close approximation
            if np.max(np.abs(old_Q-self.Q)) < epsilon:
                break

    def c_greedy_q_projection(self):
        # Initialize spibb policy to all zeros
        spibb_pol = np.zeros((self.pi_b.shape[0],self.pi_b.shape[1]))

        # TODO: Build the constrained set
        # Go through all states and actions
        for s in range(self.pi_b.shape[0]):
            # Get the unique constrained actions
            constrained_inds = np.argwhere(self.C_hat[:,0] == s).flatten()
            if len(constrained_inds) == 0:
                constrained_a = np.array([])
            else:
                constrained_a = np.unique(self.C_hat[constrained_inds][:,1])

            for a in range(self.pi_b.shape[1]):
                # If an action is known to be unsafe, we should remove all its
                # action weight
                if np.all([s,a] == self.C_hat, axis=1).any():
                    spibb_pol[s,a] = 0

                # If we do not have sufficient counts, bootstrap with baseline
                elif self.N_D[s,a] < self.N_wedge:
                    spibb_pol[s,a] = self.pi_b[s,a]

                # Perform the greedy Q projection onto approx MDP
                else:
                    # Compute the safe actions from the dataset
                    safe_a = np.argwhere(self.N_D[s,:] >= self.N_wedge).flatten()

                    # Get the actions from which we want to take the max over for
                    # Greedy Q Projection
                    proj_actions = np.unique(np.concatenate((safe_a,constrained_a),axis=0)).astype('int')

                    # Get the best action from the safe actions based on Q
                    # And assign that weight to all the other best actions
                    spibb_pol[s,np.argmax(self.Q[s,safe_a])] = np.sum(self.pi_b[s,proj_actions])
        print("RETURNING SPIBBBBBBBB")
        return spibb_pol



# Compute the Q value of the approximate MDP
def q_approx(P_hat, R_hat, gamma, policy):
    return MDP.approx_q_reward_eval(P_hat, R_hat, gamma, policy)