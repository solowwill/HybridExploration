# This file controls exploration within the Hybrid RL setting
# 
import numpy as np
import mdp as MDP

class DFE:

    def __init__(self, mdp):

        self.mdp = mdp
        self.be_c = np.zeros((self.mdp.S,self.mdp.A))

        return
    
    # Compute the exploration policy based on the DFE metric
    def pi_mcr(self, C_hat, D, N_D, N_wedge):

        self.dfe = self.compute_DFE(C_hat, D, N_D, N_wedge)

        return self.dfe / np.sum(self.dfe, axis=-1)[:,np.newaxis]
    
    # Compute the DFE function based on the belief
    def compute_DFE(self, C_hat, D, N_D, N_wedge):

        # Set C_hat and D by cutting out everything but state action pairs
        self.D = D[:,[MDP.S,MDP.A]]
        if len(C_hat) == 0:
            self.C_hat = []
        else:
            self.C_hat = C_hat[:,[MDP.S,MDP.A]]
        
        # Compute the belief distribution that a state-action is a constraint
        self.be_c = self.compute_belief(C_hat, D)

        dfe = np.zeros((self.mdp.S, self.mdp.A))

        for s in range(self.mdp.S):
            for a in range(self.mdp.A):
                if np.any([s,a] == self.D):
                    # If (s,a) is in D, we explore based on how close to 
                    # the N_wedge state count we are
                    dfe[s,a] = np.exp(1-N_D[s,a]/N_wedge) / np.e
                else:
                    # Otherwise, we explore based on belief 
                    dfe[s,a] = 1 - self.be_c[s,a]

        return dfe


    
    # Compute our belief distribution function 
    def compute_belief(self, C_hat, D):
        # Store the belief array
        be_c = np.zeros((self.mdp.S, self.mdp.A))

        # Set the constrained features
        if len(C_hat) == 0:
            constraint_feats = np.ndarray((0,MDP.FEATURE_DIMS))
        else:
            constraint_feats = self.mdp.feat_space[C_hat[:,0].astype('int'),C_hat[:,1].astype('int')]

        # Get the safe features
        safe_feats = self.mdp.feat_space[D[:,0].astype('int'),D[:,1].astype('int')]

        # Go through every state action pair
        for s in range(self.mdp.S):
            for a in range(self.mdp.A):

                # If we haven't found any constrained states yet
                if len(constraint_feats) == 0:
                    if np.any([s,a] == self.D):
                        be_c[s,a] = 0
                    else:
                        be_c[s,a] = 1

                # If we have, then compute it as a distance 
                else:
                    be_c[s,a] = 1 - np.min(np.linalg.norm(self.mdp.feat_space[s,a]-constraint_feats, axis=-1)) / \
                        np.abs(np.min(np.linalg.norm(self.mdp.feat_space[s,a]-constraint_feats, axis=-1)) - \
                        np.min(np.linalg.norm(self.mdp.feat_space[s,a]-safe_feats, axis=-1)))
                                     
        return self.be_c

    

    