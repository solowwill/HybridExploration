# Main file for all exploration agents
# Main alg class for RMAX agent
import numpy as np
from sklearn.linear_model import LinearRegression

# Main alg class for the RMAX agent
class RMAXAgent:

    def __init__(self, mdp, D=None, m=20, pib=None):
        self.name = 'rmax'
        self.mdp = mdp
        self.m = m
        self.pib = pib

        self.Q = np.zeros((self.mdp.S,self.mdp.A)) + self.mdp.rmax / (1-self.mdp.gamma)
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
            if self.n_sa[int(sarcs[utils.S]), int(sarcs[utils.A])] < self.m:

                self.n_sas[int(sarcs[utils.S]), int(sarcs[utils.A]), int(sarcs[utils.S_P])] += 1
                self.n_sa[int(sarcs[utils.S]), int(sarcs[utils.A])] +=1

                if self.r.shape == 3:
                    self.r[int(sarcs[utils.S]), int(sarcs[utils.A]), int(sarcs[utils.S_P])] += sarcs[utils.R]
                else: 
                    self.r[int(sarcs[utils.S]), int(sarcs[utils.A])] += sarcs[utils.R]
            
            # Update total seen counts
            self.NSAS[int(sarcs[utils.S]), int(sarcs[utils.A]), int(sarcs[utils.S_P])] += 1
            self.NSA[int(sarcs[utils.S]), int(sarcs[utils.A])] +=1

        # Update transitions and reward 
        self.P_hat = self.n_sas / np.where(self.n_sa > 0, self.n_sa, 1)[:,:,np.newaxis]

        if len(self.r.shape) == 3:
            self.R_hat = self.r / np.where(self.n_sas > 0, self.n_sas, 1)
        else:
            self.R_hat = self.r / np.where(self.n_sa > 0, self.n_sa, 1)

# Main alg class for the bonus RMAX agent
class bonusRMAXAgent:

    def __init__(self, mdp, D=[], m=20, pib=None):
        self.name = 'bonusrmax'
        self.mdp = mdp
        self.m = m
        self.pib = pib

        self.Q = np.zeros((self.mdp.S,self.mdp.A)) + self.mdp.rmax / (1-self.mdp.gamma)
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
            
            self.n_sas[int(sarcs[utils.S]), int(sarcs[utils.A]), int(sarcs[utils.S_P])] += 1
            self.n_sa[int(sarcs[utils.S]), int(sarcs[utils.A])] +=1

            if self.r.shape == 3:
                self.r[int(sarcs[utils.S]), int(sarcs[utils.A]), int(sarcs[utils.S_P])] += sarcs[utils.R]
            else: 
                self.r[int(sarcs[utils.S]), int(sarcs[utils.A])] += sarcs[utils.R]
            
        # Update transitions and reward 
        self.P_hat = self.n_sas / np.where(self.n_sa > 0, self.n_sa, 1)[:,:,np.newaxis]

        if len(self.r.shape) == 3:
            self.R_hat = self.r / np.where(self.n_sas > 0, self.n_sas, 1)
        else:
            self.R_hat = self.r / np.where(self.n_sa > 0, self.n_sa, 1)

# Main alg class for the bonus RMAX agent
class bonusFQIRMAXAgent:

    def __init__(self, mdp, D=[], m=20, pib=None):
        self.name = 'bonusrmax'
        self.mdp = mdp
        self.m = m
        self.pib = pib
        self.D = D

        self.Q = np.zeros((self.mdp.S,self.mdp.A)) + self.mdp.rmax / (1-self.mdp.gamma)
        self.r = np.zeros(self.mdp.R.shape)
        self.n_sa = np.zeros((self.mdp.S,self.mdp.A))
        self.n_sas = np.zeros((self.mdp.S,self.mdp.A,self.mdp.S))

        self.R_hat = np.zeros(self.mdp.R.shape)
        self.P_hat = np.zeros(self.mdp.P.shape)

        self.W = np.zeros((self.mdp.state_space.shape[1], self.mdp.action_space.shape[1]))
        self.wQ = np.zeros((self.mdp.S,self.mdp.A))

        for trajectory in D:
            self.update_counts(trajectory)
    
    def run(self, epsilon=.01):

        # Perform linear regression to find Q function
        W_old = -np.copy(self.W)
        buffer = self.make_buffer()
        x_train = self.make_train_set(buffer)
        targets = self.make_targets(buffer)
        regr = LinearRegression()

        for i in range(100):
            regr.fit(x_train, targets)
            self.W = np.reshape(regr.coef_, (self.W.shape), order='F')
            if np.sum(np.abs(self.W-W_old)) < epsilon:
                break
            W_old = np.copy(self.W)

        # Compute fitted Q function
        self.wQ = ((self.mdp.state_space @ self.W) @ self.mdp.action_space.T)

        # Compute the uncertainty over states
        u = np.maximum(0, 1 - (self.n_sa/self.m) )

        # Compute the RMAX bonus policy based on uncertainty
        self.Q[self.n_sa < self.m] = self.mdp.rmax / (1-self.mdp.gamma) + (self.wQ*u)[self.n_sa < self.m] / (1-self.mdp.gamma)

        best_actions = (self.Q - np.max(self.Q, axis=1)[:,np.newaxis]) == 0
        rmax_pol = best_actions * (1 / np.sum( best_actions, axis=1)[:,np.newaxis])

        traj, returns = self.mdp.sample_trajectory(rmax_pol)

        self.update_counts(traj)

        # Recompute Q and the opitimal policy
        pi_star, self.Q = utils.approx_q_improvement(self.mdp, self.P_hat, self.R_hat)

        return {'nsas':np.copy(self.n_sas), 'nsa':np.copy(self.n_sa), 'phat':np.copy(self.P_hat), 'rhat':np.copy(self.R_hat), 'pistar':np.copy(pi_star)}
         

    def update_counts(self, trajectory):
        self.D = np.concatenate((self.D, trajectory[np.newaxis,:,:]),axis=0)
        for sarcs in trajectory:
            
            self.n_sas[int(sarcs[utils.S]), int(sarcs[utils.A]), int(sarcs[utils.S_P])] += 1
            self.n_sa[int(sarcs[utils.S]), int(sarcs[utils.A])] +=1

            if self.r.shape == 3:
                self.r[int(sarcs[utils.S]), int(sarcs[utils.A]), int(sarcs[utils.S_P])] += sarcs[utils.R]
            else: 
                self.r[int(sarcs[utils.S]), int(sarcs[utils.A])] += sarcs[utils.R]
            
        # Update transitions and reward 
        self.P_hat = self.n_sas / np.where(self.n_sa > 0, self.n_sa, 1)[:,:,np.newaxis]

        if len(self.r.shape) == 3:
            self.R_hat = self.r / np.where(self.n_sas > 0, self.n_sas, 1)
        else:
            self.R_hat = self.r / np.where(self.n_sa > 0, self.n_sa, 1)

    # Return the state space observations
    def make_buffer(self):
        buffer = []
        for i in range(len(self.D)):
            for j in range(len(self.D[i])):
                buffer.append([self.mdp.state_space[int(self.D[i][j][utils.S])], self.mdp.action_space[int(self.D[i][j][utils.A])], \
                    self.D[i][j][utils.R], self.mdp.state_space[int(self.D[i][j][utils.S_P])]])

        return buffer

    # Make the target values for the linear regression
    def make_targets(self, buffer):
        targets = np.zeros((len(buffer), 1))

        for i in range(len(buffer)):
            _, _, r, s_p = buffer[i]
            # Compute the best action from s'
            a = np.argmax([((s_p @ self.W) @ self.mdp.action_space[j]) for j in range(self.mdp.action_space.shape[0])])

            # Compute the target to be optimized
            targets[i,-1] = r + self.mdp.gamma * ((s_p @ self.W) @ self.mdp.action_space[a])

        return targets

    # Make the training set for linear regression
    # Code borrowed from Mountain Car with Bilinear Model
    def make_train_set(self, buffer):
        num_examples = len(buffer)
        X_train = np.zeros((num_examples, self.mdp.state_space.shape[1]*self.mdp.action_space.shape[1]))
        for i in range(num_examples):
            s, a, _, _ = buffer[i]
            temp = np.zeros((1, s.shape[0]))
            temp[0,:] = s
            s = temp
            temp = np.zeros((a.shape[0],1))
            temp[:,0] = a
            a = temp
            x = np.matmul(a, s)
            x = np.append(x[0,:], x[1,:])
            X_train[i,:] = x
        return X_train
    
# Main alg class for exploration with Pi_b
class spiPiBAgent: 
    def __init__(self, mdp, D=[], m=20, pib=None):
        self.name = 'spiPib'
        self.mdp = mdp
        self.m = m
        self.pib = pib

        self.Q = np.zeros((self.mdp.S,self.mdp.A))
        self.r = np.zeros(self.mdp.R.shape)
        self.n_sa = np.zeros((self.mdp.S,self.mdp.A))
        self.n_sas = np.zeros((self.mdp.S,self.mdp.A,self.mdp.S))

        self.R_hat = np.zeros(self.mdp.R.shape)
        self.P_hat = np.zeros(self.mdp.P.shape)

        for trajectory in D:
            self.update_counts(trajectory)
    

    def run(self):

        traj, returns = self.mdp.sample_trajectory(self.pib)

        self.update_counts(traj)

        # Recompute Q and the opitimal policy
        pi_star, self.Q = utils.approx_q_improvement(self.mdp, self.P_hat, self.R_hat)

        return {'nsas':np.copy(self.n_sas), 'nsa':np.copy(self.n_sa), 'phat':np.copy(self.P_hat), 'rhat':np.copy(self.R_hat), 'pistar':np.copy(pi_star)}
         

    def update_counts(self, trajectory):

        for sarcs in trajectory:
            
            self.n_sas[int(sarcs[utils.S]), int(sarcs[utils.A]), int(sarcs[utils.S_P])] += 1
            self.n_sa[int(sarcs[utils.S]), int(sarcs[utils.A])] +=1

            if self.r.shape == 3:
                self.r[int(sarcs[utils.S]), int(sarcs[utils.A]), int(sarcs[utils.S_P])] += sarcs[utils.R]
            else: 
                self.r[int(sarcs[utils.S]), int(sarcs[utils.A])] += sarcs[utils.R]
            
        # Update transitions and reward 
        self.P_hat = self.n_sas / np.where(self.n_sa > 0, self.n_sa, 1)[:,:,np.newaxis]

        if len(self.r.shape) == 3:
            self.R_hat = self.r / np.where(self.n_sas > 0, self.n_sas, 1)
        else:
            self.R_hat = self.r / np.where(self.n_sa > 0, self.n_sa, 1)

# Main alg class for exploration with spibb
class spiSPIBBAgent: 
    def __init__(self, mdp, D=[], m=20, pib=None):
        self.name = 'spibb'
        self.mdp = mdp
        self.m = m
        self.pib = pib

        self.Q = np.zeros((self.mdp.S,self.mdp.A)) + self.mdp.rmax
        self.r = np.zeros(self.mdp.R.shape)
        self.n_sa = np.zeros((self.mdp.S,self.mdp.A))
        self.n_sas = np.zeros((self.mdp.S,self.mdp.A,self.mdp.S))

        self.R_hat = np.zeros(self.mdp.R.shape)
        self.P_hat = np.zeros(self.mdp.P.shape)


        for trajectory in D:
            self.update_counts(trajectory)
    
    def run(self,epsilon=.01):

        for _ in range(100):
            old_Q = self.Q

            spibb_pol = self.greedy_q_projection()

            self.Q = utils.approx_q_eval(self.mdp, self.P_hat, self.R_hat, spibb_pol)

            self.V = np.sum(spibb_pol * self.Q,axis=-1)

            if np.max(np.abs(old_Q-self.Q)) < epsilon:
                break
        
        traj, returns = self.mdp.sample_trajectory(spibb_pol)

        self.update_counts(traj)

        # Recompute Q and the opitimal policy
        pi_star, self.Q = utils.approx_q_improvement(self.mdp, self.P_hat, self.R_hat)

        return {'nsas':np.copy(self.n_sas), 'nsa':np.copy(self.n_sa), 'phat':np.copy(self.P_hat), 'rhat':np.copy(self.R_hat), 'pistar':np.copy(pi_star)}
         

    def update_counts(self, trajectory):

        for sarcs in trajectory:
            
            self.n_sas[int(sarcs[utils.S]), int(sarcs[utils.A]), int(sarcs[utils.S_P])] += 1
            self.n_sa[int(sarcs[utils.S]), int(sarcs[utils.A])] +=1

            if self.r.shape == 3:
                self.r[int(sarcs[utils.S]), int(sarcs[utils.A]), int(sarcs[utils.S_P])] += sarcs[utils.R]
            else: 
                self.r[int(sarcs[utils.S]), int(sarcs[utils.A])] += sarcs[utils.R]
            
        # Update transitions and reward 
        self.P_hat = self.n_sas / np.where(self.n_sa > 0, self.n_sa, 1)[:,:,np.newaxis]

        if len(self.r.shape) == 3:
            self.R_hat = self.r / np.where(self.n_sas > 0, self.n_sas, 1)
        else:
            self.R_hat = self.r / np.where(self.n_sa > 0, self.n_sa, 1)


    # Find the SPIBB optimal policy in the MLE MDP
    def greedy_q_projection(self):

        spibb_pol = np.zeros(self.pib.shape)

        for s in range(self.pib.shape[0]):
            for a in range(self.pib.shape[1]):

                # Bootstrap with baseline policy in uncertain states
                if self.n_sa[s,a] < self.m:
                        spibb_pol[s,a] = self.pib[s,a]

            # Perform the greedy Q projection onto approx MDP
            safe_a = np.argwhere(self.n_sa[s,:] >= self.m).flatten()
            if len(safe_a) != 0:
                spibb_pol[s,safe_a[np.argmax(self.Q[s,safe_a])]] = np.sum(self.pib[s,safe_a])
                
        return spibb_pol

# Main alg class for exploration with FQI
# assumes Bilinear Q function model
class FQIAgent:

    def __init__(self, mdp, D=[], m=20, pib=None):
        self.name = 'spibb'
        self.mdp = mdp
        self.m = m
        self.pib = pib
        self.D = D

        self.Q = np.zeros((self.mdp.S,self.mdp.A)) + self.mdp.rmax
        self.r = np.zeros(self.mdp.R.shape)
        self.n_sa = np.zeros((self.mdp.S,self.mdp.A))
        self.n_sas = np.zeros((self.mdp.S,self.mdp.A,self.mdp.S))

        self.W = np.zeros((self.mdp.state_space.shape[1], self.mdp.action_space.shape[1]))

        self.R_hat = np.zeros(self.mdp.R.shape)
        self.P_hat = np.zeros(self.mdp.P.shape)

        for trajectory in D:
            self.update_counts(trajectory)
    
    def run(self,epsilon=.01):

        # Perform linear regression to find Q function
        W_old = -np.copy(self.W)
        buffer = self.make_buffer()
        x_train = self.make_train_set(buffer)
        targets = self.make_targets(buffer)
        regr = LinearRegression()

        for i in range(100):
            regr.fit(x_train, targets)
            self.W = np.reshape(regr.coef_, (self.W.shape), order='F')
            if np.sum(np.abs(self.W-W_old)) < epsilon:
                break
            W_old = np.copy(self.W)

        fqi_pol = self.make_policy()

        traj, returns = self.mdp.sample_trajectory(fqi_pol)

        self.update_counts(traj)

        # Recompute Q and the opitimal policy
        pi_star, self.Q = utils.approx_q_improvement(self.mdp, self.P_hat, self.R_hat)

        return {'nsas':np.copy(self.n_sas), 'nsa':np.copy(self.n_sa), 'phat':np.copy(self.P_hat), 'rhat':np.copy(self.R_hat), 'pistar':np.copy(pi_star)}
         

    def update_counts(self, trajectory):
        self.D = np.concatenate((self.D, trajectory[np.newaxis,:,:]),axis=0)
        for sarcs in trajectory:
            
            self.n_sas[int(sarcs[utils.S]), int(sarcs[utils.A]), int(sarcs[utils.S_P])] += 1
            self.n_sa[int(sarcs[utils.S]), int(sarcs[utils.A])] +=1

            if self.r.shape == 3:
                self.r[int(sarcs[utils.S]), int(sarcs[utils.A]), int(sarcs[utils.S_P])] += sarcs[utils.R]
            else: 
                self.r[int(sarcs[utils.S]), int(sarcs[utils.A])] += sarcs[utils.R]
            
        # Update transitions and reward 
        self.P_hat = self.n_sas / np.where(self.n_sa > 0, self.n_sa, 1)[:,:,np.newaxis]

        if len(self.r.shape) == 3:
            self.R_hat = self.r / np.where(self.n_sas > 0, self.n_sas, 1)
        else:
            self.R_hat = self.r / np.where(self.n_sa > 0, self.n_sa, 1)

    # Return the state space observations
    def make_buffer(self):
        buffer = []
        for i in range(len(self.D)):
            for j in range(len(self.D[i])):
                buffer.append([self.mdp.state_space[int(self.D[i][j][utils.S])], self.mdp.action_space[int(self.D[i][j][utils.A])], \
                    self.D[i][j][utils.R], self.mdp.state_space[int(self.D[i][j][utils.S_P])]])

        return buffer

    # Make the target values for the linear regression
    def make_targets(self, buffer):
        targets = np.zeros((len(buffer), 1))

        for i in range(len(buffer)):
            _, _, r, s_p = buffer[i]
            # Compute the best action from s'
            a = np.argmax([((s_p @ self.W) @ self.mdp.action_space[j]) for j in range(self.mdp.action_space.shape[0])])

            # Compute the target to be optimized
            targets[i,-1] = r + self.mdp.gamma * ((s_p @ self.W) @ self.mdp.action_space[a])

        return targets

    # Make the training set for linear regression
    # Code borrowed from Mountain Car with Bilinear Model
    def make_train_set(self, buffer):
        num_examples = len(buffer)
        X_train = np.zeros((num_examples, self.mdp.state_space.shape[1]*self.mdp.action_space.shape[1]))
        for i in range(num_examples):
            s, a, _, _ = buffer[i]
            temp = np.zeros((1, s.shape[0]))
            temp[0,:] = s
            s = temp
            temp = np.zeros((a.shape[0],1))
            temp[:,0] = a
            a = temp
            x = np.matmul(a, s)
            x = np.append(x[0,:], x[1,:])
            X_train[i,:] = x
        return X_train

    # Make the Q function based on learned model
    def make_policy(self):

        pi = np.zeros((self.mdp.S,self.mdp.A))

        for s in range(self.mdp.S):
            a = np.argmax([((self.mdp.state_space[s] @ self.W) @ self.mdp.action_space[j]) for j in range(self.mdp.action_space.shape[0])])
            pi[s,a] = 1
        
        return pi 
    


class utils:

    S=0
    A=1
    R=2
    S_P=3   

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

        return policy, q
    
    # Perform Q-Evaluation given an MLE MDP and a policy
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