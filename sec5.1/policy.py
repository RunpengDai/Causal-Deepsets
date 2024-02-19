import numpy as np

class TopPolicy():

    def __init__(self, trajs, k):
        self.trajs = trajs
        self.k = k 
        self.mu = self.est_mu()
        
    def est_mu(self):
        states = np.array([[item[0] for item in x] for x in self.trajs])
        order = states[:,:,0]
        mu = np.mean(order, axis=1)
        return mu

    def get_action(self, states):
        # here states is a list, just keep the format
        # mu = self.est_mu()
        self.top_idx = np.argsort(-self.mu)[:self.k]
        # The idx of top k mu.
        T = len(states[0])
        N = len(states)
        

        actions = np.ones((N,T))
        sig = np.array([1 if (x in self.top_idx) else 0 for x in range(N)]).reshape(-1,1)

        actions = actions * sig
        return actions
        
        
class CarPolicy():

    def __init__(self, mod, phi):
        self.mod =mod
        self.phi = phi
        
    def logit(self,x):
        return 1/(1+np.exp(-x))
    
    def g_func(self, x, y):
        if self.mod == 'linear':
            return x + self.phi*y
        elif self.mod == 'nonlinear':
            return x + self.phi* (np.maximum(y,0)-0.63)
        elif self.mod == 'nonstationary':
            return x + self.phi*y

    def get_action(self, states):
        states = np.array(states)
        shape = list(states.shape[:-1])
        sdim = states.shape[-1]
        states = states.reshape(-1,sdim)
        U = states[:,0]
        V = states[:,1]
        logit_g = self.logit(self.g_func(V,U))
        A = np.random.binomial(1,logit_g)

        return A.reshape(shape)
class PCarPolicy():

    def __init__(self, beta, th):
        self.beta = beta
        self.th = th
        


    def get_action(self, states):



        states = np.array(states)
        shape = list(states.shape[:-1])
        sdim = states.shape[-1]
        states = states.reshape(-1,sdim)
        U = states[:,0]
        V = states[:,1]
        inpt = self.beta[0] * U + self.beta[1] * V - self.th* np.ones_like(U)
        A = np.array(inpt>0).astype(int)

        return A.reshape(shape)        
        
        
        
        
