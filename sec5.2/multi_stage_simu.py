import numpy as np
import copy
import os
import torch
from torch._C import _to_dlpack
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from modules import *
from ride_sharing_simu import * 
import distribution

class DeepDR():
    def __init__(self, seed, trajs, adj_mat, cfg):
        
        self.seed = seed
        self.cfg = cfg

        self.policy_type = cfg.CLS.policy_type
        self.gamma = cfg.CLS.gamma
    
        self.trajs = trajs

        self.adj_mat = adj_mat
        self.deepset = cfg.CLS.deepset
        self.R = len(self.trajs)
        self.N = len(self.trajs[0])
        self.T = len(self.trajs[0][0])
        self.device = cfg.CLS.device 
        self.time_tag = cfg.CLS.time_tag
        self.stag = cfg.CLS.stag
        self.normalize = cfg.CLS.normalize


        self.train_ind = np.arange(self.R)
        self.test_ind = np.arange(self.R)
        
        self.topk = cfg.CLS.topk
        self.u0s = [distribution.u_O]*self.R
        #print(self.trajs.shape)
        tgt_pi = TopPolicy(None, self.topk, 
                           distribution.u_O)
        states = np.zeros((self.R, self.N, self.T+cfg.DATA.burn_in))
        self.a_tgt = tgt_pi.get_action(states)
        
        self.dim_S = len(self.trajs[0][0][0][0]) #[o,d,m]=3
        self.dim_A = 1
        #self.deep_dim = cfg.TRAIN.deep_dim


    
    def est_reward(self):
        self.value_func = FQE_module(self.trajs, self.adj_mat,
                self.u0s, self.cfg)
        status = self.value_func.train()
        return status

            
    def est_density_ratio(self):
        if self.cfg.CLS.mod == 'RKHS':
            self.omega_func = Density_Ratio_RKHS(self.trajs, self.adj_mat,self.u0s,
                                                 self.value_func, self.cfg)
            self.omega_func.train()

        
    
    def construct_est(self):
        dr_est = 0
        is_est = 0
        for t in tqdm(range(self.T-1), mininterval=10, desc='Constructing IS estimate'):
            if self.stag == -1:
                batch_idx = np.arange(0+t,self.N*self.T*self.R+t,self.T)
            else:
                batch_idx = np.arange(self.T*self.stag+t,self.N*self.T*self.R+t,self.T*self.N)
            indexs = Batch_idx(self.R, self.N, self.T, len(batch_idx), Time=None, batch_idx = batch_idx)
            Batch, Batch_ = Batch_generator(indexs, self.trajs, self.adj_mat, self.a_tgt, self.device, "tensor",self.time_tag)
            x, _, r, neigh_x, _, ts = Batch
            x_, neigh_x_, ts_ = Batch_
            omega = self.omega_func.get_omega_value(x, neigh_x, ts).reshape(indexs.batch_size, -1)
            q = self.value_func.get_value(x, neigh_x, ts).reshape(indexs.batch_size, -1)
            if self.time_tag== True: # cheking Q_t
                print("Q{}=".format(t+1), torch.sum(q)/self.R/self.N*(1-self.gamma))
            next_qpi = self.value_func.get_value(x_, neigh_x_, ts_).reshape(indexs.batch_size, -1)
            dr_est += torch.sum((r.reshape(indexs.batch_size, -1) + self.gamma * next_qpi - q) * omega)/(self.R*(self.T-1)*self.N)
            is_est += torch.sum(omega*r) / (self.R * (self.T-1) *self.N)

        plg_est = self.construct_plg_est()
        dr_est = dr_est + plg_est
        return dr_est, is_est, plg_est

    def construct_plg_est(self):
        plg_est = 0
        if self.stag == -1:
            batch_idx = np.arange(0,self.N*self.T*self.R,self.T)
        else:
            # batch_idx = np.arange(0,self.N*self.T*self.R,self.T)
            batch_idx = np.arange(self.T*self.stag,self.N*self.T*self.R,self.T*self.N)

        # The indexs of all t=1
        indexs = Batch_idx(self.R, self.N, self.T, len(batch_idx), Time=None, batch_idx = batch_idx)
        Batch , _ = Batch_generator(indexs,[self.trajs[i] for i in self.test_ind], self.adj_mat, self.a_tgt, self.device, "tensor",self.time_tag)
        _, x, _, _, neigh_x, ts = Batch
        #print(x)
        rew_pi = self.value_func.get_value(x, neigh_x, ts).reshape(indexs.batch_size,-1)

        if self.normalize:
            rsup = self.min_max[1]
            rew_pi = rew_pi*(rsup[1]-rsup[0])+rsup[0]

        plg_est += torch.sum(rew_pi)/self.R/self.N*(1-self.gamma)
        print('Plg estimator value is {}'.format(plg_est))
        return plg_est