# This file computes everything we need for constraint inference
import numpy as np
import cvxpy as cp
import mdp as MDP

class MCR():

    def __init__(self, mdp):

        # Store the mdp
        self.mdp = mdp
    
    # Compute the regret with respect to the baseline policy and 
    def compute_C_hat(self, D, C_hat):

        # Store known constraints and known safe states
        if len(C_hat) == 0:
            constrained_feats = np.ndarray((0,MDP.FEATURE_DIMS))
        else:
            self.C_hat = C_hat[:,[MDP.S,MDP.A]]
            constrained_feats = self.mdp.feat_space[self.C_hat[:,0].astype('int'),self.C_hat[:,1].astype('int')]

        # Store the safe pairs from the dataset
        self.D = D[:,[MDP.S,MDP.A]]
        
        # Get the features of the constrained states and actions
        safe_feats = self.mdp.feat_space[self.D[:,0].astype('int'),self.D[:,1].astype('int')]
        

        # Maximize c_hat
        # Such that:
        # Norm of c_hat is less than or equal to 1
        # All safe features * c_hat are less than 0
        # All true constraints are greater than 0 
        c_hat, prob = self.construct_problem(constrained_feats, safe_feats)

        # Solve the problem 
        result = prob.solve(solver=cp.ECOS)

        # Optimal result is stored in c_hat.value 
        return c_hat.value
    

    # Define the problem instance 
    def construct_problem(self, constrained_feats, safe_feats):
        # Create the constraint funtion for optimization
        c_hat = cp.Variable(MDP.FEATURE_DIMS)
        
        # Maximize the sum of the constrained feats
        feats = np.reshape(self.mdp.feat_space, (-1, MDP.FEATURE_DIMS))
        objective = cp.Maximize(cp.sum(feats @ c_hat))

        # Subject to the problem constraints
        # No known safe feats may be labelled as a constraint
        # The norm of the constraint function must be less than or equal to 1 
        # Every known constrained feature is labelled as a constraint 
        constraints = [cp.max((safe_feats @ c_hat)) <= 0]

        return c_hat, cp.Problem(objective, constraints)
    

