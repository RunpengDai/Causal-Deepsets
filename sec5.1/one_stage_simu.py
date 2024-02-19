import numpy as np
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import *

# If I add something here.
class DeepDR():
    def __init__(self, seed, trajs, adj_mat, normalize, deepset, pi_tgt, deep_dim):
        self.seed = seed
        self.trajs_ori = trajs
        self.adj_mat = adj_mat
        self.deepset = deepset
        self.N = len(self.trajs_ori)
        self.T = len(self.trajs_ori[0])

        self.normalize = normalize
        if self.normalize:
            self.trajs, self.min_max = self.preprocess(self.trajs_ori)  # self.miin_max is a list containing
                                                                        # suprema of states and rewards
        else:
            self.trajs = self.trajs_ori
        


        self.pi_tgt = pi_tgt
        
        states = np.array([[item[0] for item in x] for x in self.trajs])
        self.dim_S = states.shape[-1]
        self.dim_A = 1
        self.deep_dim = deep_dim



    def preprocess(self, trajs):
        N = len(trajs)
        T = len(trajs[0])

        post_trajs = copy.deepcopy(trajs)
        states = np.array([[item[0] for item in x] for x in post_trajs])
        rewards = np.array([[item[2] for item in x] for x in post_trajs])
        actions = np.array([[item[1] for item in x] for x in post_trajs])

        states_s = np.min(states, axis=(0,1))
        states_m = np.max(states, axis=(0,1))
        rewards_s = np.min(rewards, axis=(0,1))
        rewards_m = np.max(rewards, axis=(0,1))
        post_states = (states - states_s[None,None,:])/(states_m[None,None,:]-states_s[None,None,:])
        post_rewards = (rewards - rewards_s)/(rewards_m - rewards_s)
        data = []
        for i in range(N):
            data_i = []
            for t in range(T):
                data_i.append([post_states[i,t], actions[i, t], post_rewards[i, t]])
            data.append(data_i)

        return data , [[states_s,states_m],[rewards_s, rewards_m]]
    
    def est_reward(self, batch_size,max_iters, hidden_dim, print_freq, lr, device ):
        input_dim = self.dim_S + self.dim_A
        self.value_func = FQE_module(self.trajs, self.adj_mat, self.deepset, input_dim, hidden_dim, self.deep_dim, lr, device )
        # model_pth = 'model/deepset_{}hdim{}_ddim{}_lr{}_iter{}.pkl'.format(self.deepset, hidden_dim,self.deep_dim,lr,max_iters)

        model_pth = 'car_model/normalize_{}deepset_{}hdim{}_ddim{}_lr{}_iter{}.pkl'.format(self.normalize, self.deepset, hidden_dim,self.deep_dim,lr,max_iters)
        # if not os.path.exists(model_pth):
        self.value_func.train(batch_size, max_iters, print_freq)
            # torch.save(self.value_func.model.state_dict(),model_pth)
        # else:
        #     print('LOADING FORMER MODEL PARMS {}...'.format(model_pth))
        #     self.value_func.model.load_state_dict(torch.load(model_pth))
        #     print('DONE!')

        
    def est_prop_score(self, rep, beh_pi, mod, thresh, eql_thresh, noise, std):
        prop_estimator = Prop_module( self.trajs, self.adj_mat, self.pi_tgt, beh_pi, thresh, eql_thresh)
        self.batch_idx, self.prob = prop_estimator.cal_prob(self.value_func, mod, rep, noise, std)
#         print(self.prob[:10])
#         print(self.batch_idx[:10])
        self.a_tgt = prop_estimator.a_tgt
        
    def check_reward(self, batch_size, rep, eql_thresh):
        per_count = 0
        noise_count = 0
        mse = 0
        for i in range(rep):
            # np.random.seed(self.seed+1)
            batch_idx = np.random.choice(self.N*self.T, batch_size, replace=False)
            s,x,r,neigh_s,neigh_x = batch_generator(batch_idx, self.trajs, self.adj_mat)
            rew_ori = self.value_func.get_value(x,neigh_x).reshape(batch_size,-1)
            if self.normalize:
                rsup = self.min_max[1]
                r = r *(rsup[1]-rsup[0])+rsup[0]
                rew_ori = rew_ori*(rsup[1]-rsup[0])+rsup[0]
            print('estimated rewrd is {}'.format(rew_ori))
            print('original reward is {}'.format(r))
            print('mse in this batch is {}'.format(np.mean(np.square(r-rew_ori))))
            mse += np.mean(np.square(r-rew_ori))
            # test reward permutation-invariant property
            permuted_neigh = permute_neigh(neigh_x)
            rew_per = self.value_func.get_value(x,permuted_neigh).reshape(batch_size,-1)
            if self.normalize:
                rew_per = rew_per*(rsup[1]-rsup[0])+rsup[0]
            print('permuted reward is {}'.format(rew_per))
            if 0:
            # if self.deepset:
                # test deepsets permutation-invariant property
                deep_ori = self.value_func.get_deepsets_result(neigh_x).reshape(batch_size,-1)
                deep_per = self.value_func.get_deepsets_result(permuted_neigh).reshape(batch_size,-1)
                print('original deepsets out is {}'.format(deep_ori))
                print('permuted deepsets out is {}'.format(deep_per))
                per_count += np.sum(equal_m(deep_per,deep_ori,eql_thresh))

                # test deepsets stability
                noise_neigh = apply_noise(neigh_x, 10)
                deep_noise = self.value_func.get_deepsets_result(noise_neigh).reshape(batch_size,-1)
                print('noised deepsets out is {}'.format(deep_noise))

                noise_count += np.sum(equal_m(deep_noise,deep_ori,eql_thresh))
        print('mean mse is {}'.format(mse/rep))
        # if self.deepset:
        if 0:
            print('permutation acc is {}, while noise acc is {}'.format(per_count/(rep*batch_size),noise_count/(rep*batch_size)))
        
    def check_prop_score(self, rep, mod, beh_pi, thresh, eql_thresh_list, noise, std ):
        
        batch_len = []
        dr_list = []
        is_list = []
        prob_list = []
        for eps in eql_thresh_list:
            prop_estimator = Prop_module( self.trajs, self.adj_mat, self.pi_tgt, beh_pi, thresh, eps)
            batch_idx, prob = prop_estimator.cal_prob(self.value_func, mod, rep, noise, std)
            a_tgt = prop_estimator.a_tgt
            print(len(batch_idx))
            print(prob[:10])
            print(np.min(prob))
            batch_len.append(len(batch_idx))
            prob_list.append(prob)
            dr_est = self.construct_dr_est(batch_idx, prob, a_tgt)
            is_est = self.construct_is_est(batch_idx, prob)
            # plg_est = self.construct_plg_est(a_tgt)
            dr_list.append(dr_est)
            is_list.append(is_est)
            print(len(batch_idx))
            # plg_list.append(plg_est)
        return batch_len, dr_list, is_list, prob_list

        
    
    def construct_dr_est(self, batch_idx, prob, a_tgt):
        s,x,r,neigh_s,neigh_x = batch_generator(batch_idx, self.trajs, self.adj_mat)
        a, neigh_a = action_generator(batch_idx, a_tgt, self.adj_mat) #[[neigh_ta],...batch...,]
        comb_x = np.concatenate([s,a],axis=1)
        comb_neigh = [np.concatenate([np.append(neigh_s[i][j],neigh_a[i][j]).reshape(1,-1) for j in range(len(neigh_s[i]))],axis=0) for i in range(len(batch_idx))]

        rew_pi = self.value_func.get_value(comb_x, comb_neigh).reshape(len(batch_idx),-1)
        rew_ori = self.value_func.get_value(x,neigh_x).reshape(len(batch_idx),-1)
        # print(rew_pi[:20])
        # print(rew_ori[:20])
        if self.normalize:
            rsup = self.min_max[1]
            r = r *(rsup[1]-rsup[0])+rsup[0]
            rew_pi = rew_pi *(rsup[1]-rsup[0])+rsup[0]
            rew_ori = rew_ori*(rsup[1]-rsup[0])+rsup[0]

        weight = 1/(prob.reshape(len(batch_idx),-1))
        
        dr_est = np.sum(weight*(r-rew_ori))/self.T/self.N



        plg_est = 0
        for i in range(self.N):
            batch_idx = np.arange(i*self.T,(i+1)*self.T)
            s,x,r,neigh_s,neigh_x = batch_generator(batch_idx, self.trajs, self.adj_mat)
            a, neigh_a = action_generator(batch_idx, a_tgt, self.adj_mat) #[[neigh_ta],...batch...,]
            comb_x = np.concatenate([s,a],axis=1)
            comb_neigh = [np.concatenate([np.append(neigh_s[i][j],neigh_a[i][j]).reshape(1,-1) for j in range(len(neigh_s[i]))],axis=0) for i in range(len(batch_idx))]

            rew_pi = self.value_func.get_value(comb_x, comb_neigh).reshape(len(batch_idx),-1)
            # print(rew_pi[:20])
            if self.normalize:
                rsup = self.min_max[1]
                rew_pi = rew_pi*(rsup[1]-rsup[0])+rsup[0]
            # print(rew_pi[:20])
            plg_est += np.sum(rew_pi)/self.T/self.N

        dr_est = dr_est + plg_est

        print('DR esimator value is {}'.format(dr_est))
        return dr_est

    def construct_is_est(self, batch_idx, prob):
        s,x,r,neigh_s,neigh_x = batch_generator(batch_idx, self.trajs, self.adj_mat)
        if self.normalize:
            rsup = self.min_max[1]
            r = r *(rsup[1]-rsup[0])+rsup[0]
        weight = 1/(prob.reshape(len(batch_idx),-1))

        
        is_est = np.sum(weight*r)/self.T/self.N
        print('IS esimator value is {}'.format(is_est))
        return is_est
        
    def construct_plg_est(self, a_tgt):
        plg_est = 0
        for i in range(self.N):
            batch_idx = np.arange(i*self.T,(i+1)*self.T)
            s,x,r,neigh_s,neigh_x = batch_generator(batch_idx, self.trajs, self.adj_mat)
            a, neigh_a = action_generator(batch_idx, a_tgt, self.adj_mat) #[[neigh_ta],...batch...,]
            comb_x = np.concatenate([s,a],axis=1)
            comb_neigh = [np.concatenate([np.append(neigh_s[i][j],neigh_a[i][j]).reshape(1,-1) for j in range(len(neigh_s[i]))],axis=0) for i in range(len(batch_idx))]

            rew_pi = self.value_func.get_value(comb_x, comb_neigh).reshape(len(batch_idx),-1)
            # print(rew_pi[:20])
            if self.normalize:
                rsup = self.min_max[1]
                rew_pi = rew_pi*(rsup[1]-rsup[0])+rsup[0]
            # print(rew_pi[:20])
            plg_est += np.sum(rew_pi)/self.T/self.N
        print('Plg estimator value is {}'.format(plg_est))
        return plg_est
        
    
            

