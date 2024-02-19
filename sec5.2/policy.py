import numpy as np
import distribution

class TopPolicy():
    def __init__(self, trajs, k, u_0):
        self.k = k 

    def get_action(self, states):
        # here states of shape(R, N, T)
        # mu = self.est_mu()

        Rep, N, T = states.shape
        self.mu = distribution.u_O[str(int(np.sqrt(N)))]

        top_idx = np.argsort(-self.mu)[:self.k]

        # The idx of top k mu.

        sig = np.zeros((N,1))
        sig[top_idx] =1
        actions = sig.repeat(T+1, axis=1)

        return [actions]*Rep
        

        
if __name__ == "__main__":
    tgt = TopPolicy(trajs = None, k= 15, u_0=distribution.u_O)
    states = np.zeros((2, 100, 200))
    print(np.sum(tgt.get_action(states)[0],axis = 1))
        
        
        
        
