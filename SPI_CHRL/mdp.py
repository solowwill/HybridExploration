# This file allows for Safe Policy Improvement with Baseline Bootstrapping
import numpy as np
from gym_simplegrid.envs.simple_grid import SimpleGridEnv as gridenv
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import sklearn.linear_model
import gym_simplegrid


# SHOULD BE A MULTIPLE OF 4
FEATURE_DIMS = 2

# Set seed for reproducibility
SEED = 0

# Set indices of trajectory tuples 
S = 0
A = 1
R = 2
C = 3
S_P = 4

class mdp():

    def __init__(self, map, num_actions, init_state):
        
        # Convert the state space to a numpy array and create action space
        self.state_space = np.asarray(map, dtype='c').astype('int')
        self.action_space = np.arange(num_actions)

        # Build the parts of the MDP
        self.S = self.state_space.shape[0] * self.state_space.shape[1]
        self.A = num_actions
        self.P = self.build_dynamics()
        self.R = self.build_reward()
        self.C = self.build_constraints()
        self.gamma = .9

        # Create the feature space with labels, and a classifying constraint function
        self.feat_space, self.feat_labels, self.constraint_function  = self.create_feature_space()

        # Define the goal state and init state
        self.goal_state = np.argwhere(self.state_space==gridenv.GOAL).flatten()
        self.init_state = init_state
        self.curr_state = self.to_s(self.init_state)

        # Set a max horizon
        self.H = 50


    def sample_trajectory(self, policy):

        # Initialize the sar array to a starting value
        sarcs = []

        # Reset initial state
        self.curr_state = self.to_s(self.init_state)

        # Run while the mdp is not in a goal state or for a finite number of episodes
        while np.any(self.curr_state != self.to_s(self.goal_state)):

            # Get the next action for the mdp based on the policy
            curr_action = np.random.choice(np.arange(self.A), 1, p=policy[self.curr_state,:])[0]

            # Get the next state in the mdp based on the action and dymamics)
            next_state = np.random.choice(np.arange(self.S), 1, p=self.P[self.curr_state,curr_action,:])[0]

            reward = self.R[self.curr_state, curr_action]
            constraint = self.C[self.curr_state, curr_action]

            # Store the SAR for graphing
            sarcs.append( [self.curr_state, curr_action, reward, constraint, next_state] )

            # Stop MDP if running for x amount of states
            if len(sarcs) >= self.H:
                break

            # Update state
            self.curr_state = next_state
            
        return np.array(sarcs)
        

    # Input: The state space and the action space
    # Output: S x A x S array of dynamics
    def build_dynamics(self):

        # Record the x length for indexing
        x = self.state_space.shape[1]

        map = np.copy(self.state_space)

        # Replace any non-wall instances with free space
        map[map != gridenv.WALL] = gridenv.FREE
        map = map.flatten()

        # Create the dynamics array
        dynamics = np.zeros((map.shape[0], self.A, map.shape[0]))

        # Go through every state in the dynamics
        for i in range(dynamics.shape[0]):
            
            # If there is a wall, no action will take us out of this state
            # So, probability of staying in that state is 1
            if map[i] == gridenv.WALL:
                dynamics[i,:,i] = 1

            # Now, if the state is a inhabitable state by the agent, or water, we should 
            # allow the agent to move from it
            if (map[i] == gridenv.FREE):

                # NOTE actions {0, 1, 2, 3} = {up, down, left, right}
                # If there is a state in the grid above the current state and it is not a wall
                # then we can move up to it, otherwise stay in the same state
                if (i - x) >= 0:
                    
                    if map[i-x] != gridenv.WALL:
                        dynamics[i,0,i-x] = 1
                    else: 
                        dynamics[i,0,i] = 1
                else: 
                    dynamics[i,0,i] = 1
                
                # If there is a state in the grid below the current state and it is not a wall
                # Then we can move down to it 
                if (i + x) < dynamics.shape[0]:
                    if map[i+x] != gridenv.WALL:
                        dynamics[i,1,i+x] = 1
                    else:
                        dynamics[i,1,i] = 1
                else:
                    dynamics[i,1,i] = 1

                # If there is a state to the left of the current state and it is not a wall
                # Then we can move left to it 
                if (i-1) % x != (x-1):
                    if map[i-1] != gridenv.WALL:
                        dynamics[i,2,i-1] = 1
                    else: 
                        dynamics[i,2,i] = 1
                else: 
                    dynamics[i,2,i] = 1

                # If there is a state to the right of the current state and it is not a wall
                # Then we can move right to it 
                if (i+1) % x != 0:
                    if map[i+1] != gridenv.WALL:
                        dynamics[i,3,i+1] = 1
                    else: 
                        dynamics[i,3,i] = 1
                else: 
                    dynamics[i,3,i] = 1

        return dynamics

    # Input: The state space and the action space
    # Output: S x A array of rewards
    def build_reward(self):

        map = self.state_space.flatten()
        
        # Create the rewards array
        rewards = np.zeros((self.S, self.A))

        # Go through every state in the rewards
        for i in range(rewards.shape[0]):
            # Go through every action in the rewards
            for j in range(rewards.shape[1]):
                next_states = np.argwhere(self.P[i,j] != 0)
                for ns in next_states:
                    #  moving into goal state is 10
                    if map[ns] == gridenv.GOAL:
                        rewards[i,j] = 10
                    # Reward for movement is -1
                    else:
                        rewards[i,j] = -1     
        return rewards
    
    # Input: The state space and the action space
    # Output: S x A array of constraints
    def build_constraints(self):

        map = self.state_space.flatten()
        
        # Create the constraints array
        constraints = np.zeros((self.S, self.A))

        # Go through every state in the constraints
        for i in range(constraints.shape[0]):
            # Go through every action in the constraints
            for j in range(constraints.shape[1]):
                next_states = np.argwhere(self.P[i,j] != 0)
                for ns in next_states:
                    # if the agent moves over lava
                    if map[ns] == gridenv.LAVA:
                        constraints[i,j] = 10
                    # Otherwise, no constraint violation
                    else:
                        constraints[i,j] = 0
        return constraints
    
    # Create the feature space as clusters 
    # TODO fix the mapping of state features to state action features
    def create_feature_space(self):
        # Get the number of safe and constraint states
        num_constraints = np.sum(self.C > 0)
        num_safe = np.sum(self.C <= 0)

        # Create the centers
        centers, centers_std = self.create_centers()

        # Make the clusters of features
        '''c_feats, _ = make_blobs(n_samples=num_constraints, centers=centers[:int(FEATURE_DIMS/2)], \
                                cluster_std=centers_std[:int(FEATURE_DIMS/2)],n_features=FEATURE_DIMS, random_state=SEED)
        s_feats, _ = make_blobs(n_samples=num_safe, centers=centers[int(FEATURE_DIMS/2):], \
                                cluster_std=centers_std[int(FEATURE_DIMS/2):],n_features=FEATURE_DIMS, random_state=SEED)'''
        
        # Proof of concept for small features 
        # TODO Comment this out 
        c_feats, _ = make_blobs(n_samples=num_constraints, centers=[[1,1]],cluster_std=[[.5]],n_features=2)
        s_feats, _ = make_blobs(n_samples=num_safe, centers=[[-1,-1]], cluster_std=[[.5]],n_features=2)
    
        # Create labels for features
        c_y = np.ones(num_constraints)
        s_y = np.zeros(num_safe)

        # Concatenate all features and labels together 
        feats = np.concatenate((c_feats,s_feats),axis=0)
        labels = np.concatenate((c_y,s_y),axis=0)
        
        norm_factor = np.max(np.linalg.norm(feats,axis=1))
        # Normalize the features to norm 1
        feats = feats / norm_factor

        # Perform logistic regression without the bias term so we get a dot product
        log_reg = sklearn.linear_model.LogisticRegression(penalty='l2',solver='liblinear', fit_intercept=False)
        log_reg.fit(feats, labels)

        # Map the feature space into the map
        feature_map = np.zeros((self.C.shape[0], self.C.shape[1], FEATURE_DIMS))
        feature_labels = np.zeros((self.C.shape[0],self.C.shape[1]))

        c_args = np.argwhere(self.C > 0)
        s_args = np.argwhere(self.C <= 0)

        for i in range(c_args.shape[0]):
            feature_map[c_args[i,0],c_args[i,1]] = c_feats[i] / norm_factor
            feature_labels[c_args[i,0],c_args[i,1]] = c_y[i]
        for i in range(s_args.shape[0]):
            feature_map[s_args[i,0],s_args[i,1]] = s_feats[i] / norm_factor
            feature_labels[s_args[i,0],s_args[i,1]] = s_y[i]   

        coef = log_reg.coef_[0]
        x = np.arange(start=-1, stop=1, step=.1)
        plt.plot(x, -coef[1]*x)
        plt.scatter(feats[labels==0,0], feats[labels==0,1],color='red')
        plt.scatter(feats[labels==1,0], feats[labels==1,1],color='blue')
        for i in range(self.S):
            for j in range(self.A):
                plt.annotate(f'[{i},{j}]', (feature_map[i,j,0],feature_map[i,j,1]))
        plt.savefig('feats.png')

        # Return the coefficients for classification
        return feature_map/norm_factor, feature_labels, log_reg.coef_[0]

    # Create cluster centers and std
    def create_centers(self):
        # Create the number of clusters
        num_clust = int(FEATURE_DIMS / 2)

        # Create clusters on the opposing side of the hypercube
        clust1 = np.ones(shape=(num_clust, FEATURE_DIMS))
        clust2 = -1 * np.ones(shape=(num_clust, FEATURE_DIMS))

        # Create a random modification of the last half of the cluster centers 
        # To move them around on the vertices of the hypercube
        clust_update = np.random.randint(2, size=(num_clust,num_clust))
        clust_update[clust_update == 0] = -1

        # Update the clusters
        clust1[:,num_clust:] = clust_update
        clust2[:,num_clust:] = clust_update

        return np.concatenate((clust1,clust2)), .5*np.ones(FEATURE_DIMS)
    
    def to_s(self, state):
        return state[0] * self.state_space.shape[1] + state[1]
    

# Evaluate a policy for state-constraint violation
# Input: an MDP and a policy
# Output: the value function for a given state for that policy
def iter_v_constraint_eval(mdp, policy, epsilon=.01):

    # Initialize delta, the change in value accuracy
    # Initialize the value function, the goal state must be 0
    j = np.zeros(mdp.S)
    
    # Perform until a certain threshold is met
    while True:
        # Store the current value function
        old_j = j

        # Compute the new value function based on the current value function
        # Sum along reward axis and state to compute the expected reward of a state-action-state pair
        # Then add the expected value from states and multiply by policy to find value
        j = np.sum( np.sum( mdp.P * (mdp.C[:,:,np.newaxis] + mdp.gamma * j), axis=-1) * policy, axis=-1)

        # Exit the loop once sufficiently close approximation
        if np.max(np.abs(old_j-j)) < epsilon:
            break

    return j
    
# Evaluate a given policy for state-action constraint
# Input: an MDP and a policy
# Output: the value function for a given state for that policy
def iter_q_constraint_eval(mdp, policy, epsilon=.01):

    # Initialize delta, the change in value accuracy
    # Initialize the value function, the goal state must be 0
    j = np.zeros((mdp.S,mdp.A))
    
    # Perform until a certain threshold is met
    while True:
        # Store the current value function
        old_j = j

        # Compute the new value function based on the current value function
        # Sum along reward axis and state to compute the expected reward of a state-action-state pair
        # Then add the expected value from states and multiply by policy to find value
        j = mdp.C + mdp.gamma * np.sum(mdp.P * np.sum(policy * j, axis=-1), axis=-1)

        # Exit the loop once sufficiently close approximation
        if np.max(np.abs(old_j-j)) < epsilon:
            break

    return j

# Evaluate a given policy on a particular MDP
# Input: an MDP and a policy
# Output: the value function for a given state for that policy
def iter_v_reward_eval(mdp, policy, epsilon=.01):

    # Initialize delta, the change in value accuracy
    # Initialize the value function, the goal state must be 0
    v = np.zeros(mdp.S)
    
    # Perform until a certain threshold is met
    while True:
        # Store the current value function
        old_v = v

        # Compute the new value function based on the current value function
        # Sum along reward axis and state to compute the expected reward of a state-action-state pair
        # Then add the expected value from states and multiply by policy to find value
        v = np.sum( np.sum( mdp.P * (mdp.R[:,:,np.newaxis] + mdp.gamma * v), axis=-1) * policy, axis=-1)
        # Treat as an absorbing state

        # Exit the loop once sufficiently close approximation
        if np.max(np.abs(old_v-v)) < epsilon:
            break

    return v

# Evaluate a given policy on a particular MDP
# Input: an MDP and a policy
# Output: the value function for a given state for that policy
def iter_q_reward_eval(mdp, policy, epsilon=.01):

    # Initialize delta, the change in value accuracy
    # Initialize the value function, the goal state must be 0
    q = np.zeros(mdp.S)
    
    # Perform until a certain threshold is met
    while True:
        # Store the current value function
        old_q = q

        # Compute the new value function based on the current value function
        # Sum along reward axis and state to compute the expected reward of a state-action-state pair
        # Then add the expected value from states and multiply by policy to find value
        q = mdp.R + mdp.gamma * np.sum(mdp.P * np.sum(policy * q, axis=-1), axis=-1)
        # Treat as an absorbing state

        # Exit the loop once sufficiently close approximation
        if np.max(np.abs(old_q-q)) < epsilon:
            break

    return q


# Evaluate a given policy on an approximate MDP
# Input: an MDP and a policy
# Output: the value function for a given state for that policy
def approx_q_reward_eval(P_hat, R_hat, gamma, policy, epsilon=.01):

    # Initialize delta, the change in value accuracy
    # Initialize the value function, the goal state must be 0
    q = np.zeros((P_hat.shape[0],P_hat.shape[1]))
    
    # Perform until a certain threshold is met
    while True:
        # Store the current value function
        old_q = q

        # Compute the new value function based on the current value function
        # Sum along reward axis and state to compute the expected reward of a state-action-state pair
        # Then add the expected value from states and multiply by policy to find value
        q = R_hat + gamma * np.sum(P_hat * np.sum(policy * q, axis=-1), axis=-1)
        # Treat as an absorbing state

        # Exit the loop once sufficiently close approximation
        if np.max(np.abs(old_q-q)) < epsilon:
            break

    return q




# Value improvement
# Input: an MDP 
# Output: a better policy
def value_improvement(mdp, epsilon=.01):

    # Initialize delta, the change in value accuracy
    # Initialize the value function, the goal state must be 0
    v = np.zeros(mdp.S)
    
    # Perform until a certain threshold is met
    while True:
        # Store the current value function
        old_v = v

        # Compute the new value function based on the current value function
        # Sum along reward axis and state to compute the expected reward of a state-state-action pair
        # Then add the expected value from states and multiply by policy to find value
        v = np.max(np.sum( mdp.P * (mdp.R[:,:,np.newaxis] + mdp.gamma * v), axis=-1), axis=-1)

        # Exit the loop once sufficiently close approximation
        if np.max(np.abs(old_v-v)) < epsilon:
            break
            

    # Compute the probability of the best action with ties broken arbitrarily 
    # First compute the value of each of the actions
    action_reward = np.sum(mdp.P * (mdp.R[:,:,np.newaxis] + mdp.gamma * v), axis=-1)

    # Find the best actions with ties broken arbitrarily
    best_actions = (action_reward - np.max(action_reward, axis=1)[:,np.newaxis]) == 0
    policy = best_actions * (1 / np.sum( best_actions, axis=1)[:,np.newaxis])

    return policy, v


# Q improvement
# Input: an MDP 
# Output: a better policy
def q_improvement(mdp, epsilon=.01):

    # Initialize delta, the change in value accuracy
    # Initialize the value function, the goal state must be 0
    q = np.zeros((mdp.S,mdp.A))
    
    # Perform until a certain threshold is met
    while True:
        # Store the current value function
        old_q = q

        # Compute the new value function based on the current value function
        # Sum along reward axis and state to compute the expected reward of a state-state-action pair
        # Then add the expected value from states and multiply by policy to find value
        #q = np.max(np.sum( mdp.P * (mdp.R[:,:,np.newaxis] + mdp.gamma * q), axis=-1), axis=-1)
        q = np.sum(mdp.P * (mdp.R[:,:,np.newaxis] + mdp.gamma * np.max(q, axis=-1)), axis=-1)

        # Exit the loop once sufficiently close approximation
        if np.max(np.abs(old_q-q)) < epsilon:
            break
            

    # Compute the probability of the best action with ties broken arbitrarily 
    # First compute the value of each of the actions
    action_reward = np.sum(mdp.P * (mdp.R[:,:,np.newaxis] + mdp.gamma * np.max(q, axis=-1)), axis=-1)

    # Find the best actions with ties broken arbitrarily
    best_actions = (action_reward - np.max(action_reward, axis=1)[:,np.newaxis]) == 0
    policy = best_actions * (1 / np.sum( best_actions, axis=1)[:,np.newaxis])

    return policy, q
