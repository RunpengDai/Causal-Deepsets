from audioop import mul
import numpy as np
from utils import getAdjGrid


class Simulator(object):
    def __init__(self, l,  fmod, bmod,  phi, rho_u, rho_v, sigma, threshold):
        self.L = l
        self.N = l**2
        self.fmod = fmod
        self.bmod = bmod

        self.phi = phi
        self.rho_u = rho_u
        self.rho_v = rho_v
        self.sigma = sigma
        self.th = threshold

        self.get_cov_matrix()

    def logit(self,x):
        return 1/(1+np.exp(-x))
    
    def g_func(self, u, v):
        
        if self.fmod:
            return u+v
        else:
            l2 = np.square(u-v)
            sigmoid = 1/(1+np.exp(-u*v))
            return l2 +sigmoid

    
    def get_cov_matrix(self):
        self.adj_mat = W = getAdjGrid(self.L)
        M = np.diag(np.sum(W,axis=1))
        self.cov_u = np.linalg.inv(M-self.rho_u*W) * (self.sigma**2)
        self.cov_v = np.linalg.inv(M-self.rho_v*W) * (self.sigma**2)

    def get_beta_matrix(self):
        if self.bmod :
            b_mat = 2 * np.eye(self.N) - 0.5 * self.adj_mat
        else :
            multilier = 0.3* np.array([(-1)**i for i in range(self.N)]).reshape(1,-1) - 0.5
            b_mat = np.tile(multilier,(self.N,1))* self.adj_mat
            for i in range(self.N):
                b_mat[i,i] = 1.5
        return b_mat
    
    def get_max_matrix(self,x):
        '''
        input shape : (self.N, T)
        '''
        T = x.shape[-1]
        mat = np.zeros((self.N, T))
        for t in range(T):
            rep_data = x[:,t]
            for i in range(self.N):
                mat[i,t] = np.max(rep_data*self.adj_mat[i])
        return mat
        
           

    def simu_trajs(self, t):
        U = np.random.multivariate_normal(np.zeros(self.L**2),self.cov_u,(t,)).T
        V = np.random.multivariate_normal(np.zeros(self.L**2),self.cov_v,(t,)).T
        logit_g = 0.5 * np.ones_like(U)
        A = np.random.binomial(1,logit_g)
        beta = self.get_beta_matrix()

        if self.fmod:
            inpt =  0.1*np.dot(beta,A)  + np.dot(beta, U) + np.dot(beta, V)
        else:
            inpt = 0.1*np.dot(beta,A)  + np.dot(beta, A*U)
        R = np.random.randn(*inpt.shape) + inpt

   
        data = []
        for i in range(self.N):
            data_i = []
            for tt in range(t):
                data_i.append([np.array([U[i, tt], V[i, tt]]), A[i, tt], R[i, tt]])
            data.append(data_i)
        return data, self.adj_mat, [[U, V], A, R]

    def get_real_value(self, t, pi):

        U = np.random.multivariate_normal(np.zeros(self.L**2),self.cov_u,(t,)).T
        V = np.random.multivariate_normal(np.zeros(self.L**2),self.cov_v,(t,)).T
        S = np.stack([U,V]).transpose(1,2,0)
        A = pi.get_action(S)
        beta = self.get_beta_matrix()
#         if self.fmod:
#             inpt = np.dot(beta,A) +np.dot(beta,U) + np.dot(beta,V)
#         else:
#             inpt = np.dot(beta,A) +np.dot(beta,U) + np.dot(beta,V) + self.get_max_matrix(self.g_func(U+V))
#         inpt = 0.1*np.dot(beta,A) + np.dot(beta, self.g_func(U,V))
        if self.fmod:
            inpt =  0.1*np.dot(beta,A)  + np.dot(beta, U) + np.dot(beta, V)
        else:
            inpt = 0.1*np.dot(beta,A)  + np.dot(beta, A*U)
        R = inpt


        return np.sum(R)/t/self.N

    def get_tgt_trajs(self, t, pi):

        U = np.random.multivariate_normal(np.zeros(self.L**2),self.cov_u,(t,)).T
        V = np.random.multivariate_normal(np.zeros(self.L**2),self.cov_v,(t,)).T
        S = np.stack([U,V]).transpose(1,2,0)
        A = pi.get_action(S)
        beta = self.get_beta_matrix()

        # print(self.g_func(A))
        # print(beta)
#         if self.fmod:
#             inpt = np.dot(beta,A) +np.dot(beta,U) + np.dot(beta,V)
#         else:
#             inpt = np.dot(beta,A) +np.dot(beta,U) + np.dot(beta,V) + self.get_max_matrix(self.g_func(U+V))
#         inpt =  np.dot(beta,self.g_func(U)) + np.dot(beta,self.g_func(V))
#         inpt = 0.1*np.dot(beta,A) + np.dot(beta, self.g_func(U,V))
        if self.fmod:
            inpt =  0.1*np.dot(beta,A)  + np.dot(beta, U) + np.dot(beta, V)
        else:
            inpt = 0.1*np.dot(beta,A)  + np.dot(beta, A*U)
        R = np.random.randn(*inpt.shape) + inpt

   
        data = []
        for i in range(self.N):
            data_i = []
            for tt in range(t):
                data_i.append([np.array([U[i, tt], V[i, tt]]), A[i, tt], R[i, tt]])
            data.append(data_i)
        return data, self.adj_mat, [[U, V], A, R]


    



    
        




