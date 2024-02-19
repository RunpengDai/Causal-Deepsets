import copy
import numpy as np
from tqdm import tqdm
from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F


class Model_Behav(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.dim_S = input_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x 


class Reward_Estimator(nn.Module):
    def __init__(self, deepset, input_dim, hidden_dim,  deep_dim):
        super(Reward_Estimator, self).__init__()
        
        self.deepset = deepset
        
        self.feature_extractor=nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 10),
            nn.ReLU(inplace=True),
        )
        # feature extractor is for regions.
        if self.deepset:
            self.ds = Deepsets(input_dim, 32, deep_dim)
        # normal extractor is for center.
            self.output_layer = nn.Sequential(
                                            #nn.BatchNorm1d(deep_dim + 10),
                                            # nn.Linear(1 + 10, hidden_dim),
                                            nn.Linear(deep_dim + 10, hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, 1)                           
                                            )
        else:
            self.output_layer = nn.Sequential(
                                # nn.BatchNorm1d(input_dim*2),
                                nn.Linear(input_dim *2, hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, int(hidden_dim/2)),
                                nn.ReLU(inplace=True),
                                nn.Linear(int(hidden_dim/2), 1)                           
                                )

# feature ->10, ->6 133 MSE
    def forward(self, x, neigh_x):
        
        
        if self.deepset:
            x = self.feature_extractor(x)
            neigh = self.ds(neigh_x)
        else:
            neigh = torch.cat([torch.mean(x,0).view(1,-1) for x in neigh_x])
        
        out = torch.cat([x,neigh],axis=1)
        return self.output_layer(out)
        


    
class Deepsets(nn.Module):
    def __init__(self,input_dim, hidden_dim, out_dim):
        super(Deepsets, self).__init__()

        self.phi = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(inplace = True))
        self.rho = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(hidden_dim, out_dim),
                                            nn.ReLU(inplace = True))

    def forward(self, neigh_x):
        out = torch.cat([torch.sum(self.phi(x),0).view(1,-1) for x in neigh_x])
        out = self.rho(out)
        # out = torch.mean(out,1,keepdim=True)
        # print(out.shape)
        return out
    
def batch_generator(batch_idx, trajs, adj_mat):
    N = len(trajs)
    T = len(trajs[0])
    n_idx = (batch_idx/T).astype(int)
    t_idx = batch_idx - n_idx*T
    
    states = np.concatenate(
        [trajs[i][j][0].reshape(1,-1) for i,j in zip(n_idx,t_idx)],axis=0)
    actions = np.concatenate(
        [trajs[i][j][1].reshape(1,-1) for i,j in zip(n_idx,t_idx)],axis=0)
    x = np.concatenate([states, actions], axis=1)
    rewards = np.concatenate(
        [trajs[i][j][2].reshape(1,-1) for i,j in zip(n_idx,t_idx)],axis=0)
    
    neigh_s = [np.concatenate([trajs[i][t][0].reshape(1,-1) for i in np.where(adj_mat[n]==1)[0]],axis=0) for n,t in zip(n_idx,t_idx)]
    neigh_x = [np.concatenate([np.append(trajs[i][t][0],trajs[i][t][1]).reshape(1,-1) for i in np.where(adj_mat[n]==1)[0]],axis=0) for n,t in zip(n_idx,t_idx)]
    return states, x, rewards, neigh_s, neigh_x

def action_generator(batch_idx, tgt_action, adj_mat):
    N = len(tgt_action)
    T = len(tgt_action[0])
    n_idx = (batch_idx/T).astype(int)
    t_idx = batch_idx - n_idx * T
    
    a = np.concatenate(
        [tgt_action[i][j].reshape(1,-1) for i,j in zip(n_idx,t_idx)],axis=0)

    ta = [np.concatenate([tgt_action[i][t].reshape(-1,1) for i in np.where(adj_mat[n]==1)[0]],axis=0) for n,t in zip(n_idx,t_idx)]
    return a, ta

def permute_neigh(neigh_x):

    perm_neigh = [x[np.random.choice(len(x),len(x),replace=False)] for x in neigh_x]
    return perm_neigh

def apply_noise(neigh_x, noise):
    neigh = copy.deepcopy(neigh_x)
    for x in neigh:
        i = np.random.choice(x.shape[0],1)[0]
        j = np.random.choice(x.shape[1],1)[0]
        x[i][j] = x[i][j]+noise
    return neigh


def idx2batch(idx, N, T):
#     batch_idx = np.array([])
#     for i in range(N):
#         if len(idx[i])==0:
#             pass
#         else:
#             batch_idx = np.append(batch_idx, idx[i]+i*T)
    batch_idx = np.concatenate([x+i*T for i,x in enumerate(idx)])
    return batch_idx

def equal_m(v1, v2, thresh):
    # use average version of Euclid distance as the criterion of equal.
    eql = (np.sqrt(np.mean(np.square(v1-v2),axis=1)) < (thresh*np.sqrt(np.mean(np.square(v2),axis=1))))
    return eql
def wrap_traversal(x):
    # input ndarray for one point's nerghbors
    # output this point's all possible action combinations, a list
    bit = len(x)
    action_matrix = np.concatenate([np.array(list(np.binary_repr(i,bit))).astype(int).reshape(1,-1) for i in range(2**bit)],axis=0)
    neigh = [np.concatenate([x,act.reshape(-1,1)],axis=-1) for act in action_matrix]
    return neigh, np.sum(action_matrix, axis=1)




class FQE_module():
    def __init__(self, trajs, adj_mat, deepset,
                 input_dim, hidden_dim, deep_dim,
                 lr, device ):
        
        self.deepset = deepset
        self.model = Reward_Estimator(deepset, input_dim, hidden_dim, deep_dim).to(device)
        self.trajs = trajs
        self.adj_mat = adj_mat
        self.N = len(self.trajs)
        self.T = len(self.trajs[0])
        self.NT = self.T*self.N
        
        
        self.device = device
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)
#         self.scheduler = MultiStepLR(optimizer, milestones=[1000, 1700, 2500, 4000], gamma=0.3)
        
    def train(self, batch_size, max_iters, print_freq):
        
        print('Start Training Reward Function Procedure...')
        self.model.train()
        
        for i in tqdm(range(max_iters)):
            batch_idx = np.random.choice(self.NT, size=batch_size)
            s,x,r,neigh_s,neigh_x = batch_generator(batch_idx, self.trajs, self.adj_mat)
            # batch_generator not Checked yet .
            x,r = (torch.from_numpy(x.astype('float32')).to(self.device),
                   torch.from_numpy(r.astype('float32')).to(self.device))
            neigh_x = [torch.from_numpy(x.astype('float32')).to(self.device) for x in neigh_x]
            self.optimizer.zero_grad()
            pred = self.model(x,neigh_x)
            
            loss = self.loss_fn(pred, r)
            
            loss.backward()
            self.optimizer.step()
            if i% print_freq==0:
                with torch.no_grad():
                    print("Train MSE :{}".format(loss.item()))
            
    def get_value(self, x, neigh_x):
        x = torch.from_numpy(x.astype('float32')).to(self.device)
        neigh_x = [torch.from_numpy(x.astype('float32')).to(self.device) for x in neigh_x]
        pred = self.model(x,neigh_x)
        return pred.detach().cpu().numpy()
    
    def get_deepsets_result(self, neigh_x):
        neigh_x = [torch.from_numpy(x.astype('float32')).to(self.device) for x in neigh_x]
        if self.deepset:
            pred = self.model.ds(neigh_x)
        else:
            pred = torch.cat([torch.mean(x,0).view(1,-1) for x in neigh_x])
        return pred.detach().cpu().numpy()
        
            

class Prop_module():
    def __init__(self, trajs, adj_mat, pi_tgt, beh_pi, thresh, eql_thresh):
        self.trajs = trajs
        self.adj_mat = adj_mat
        self.thresh = thresh
        self.eql_thresh = eql_thresh 

        self.N = len(self.trajs)
        self.T = len(self.trajs[0])

        self.states = [[item[0] for item in x] for x in self.trajs]
        self.actions = [[item[1] for item in x] for x in self.trajs]

        self.beh_pi = beh_pi

        self.a_tgt = pi_tgt.get_action(np.array(self.states))

    def filter(self, rew_estimator):
        print('Now Filtering...')
        self.idents = []
        for i in tqdm(range(len(self.trajs))):
            batch_idx = np.arange(i*self.T,(i+1)*self.T)
            s, x, r, neigh_s, neigh_x = batch_generator(batch_idx, self.trajs, self.adj_mat)

            # Does the tgt policy need gloabal information? It seems yes.
            # So how can we get action via neighbor states?
            a, tna = action_generator(batch_idx, self.a_tgt, self.adj_mat) #[[neigh_ta],...batch...,]
            comb_sa = [np.concatenate([np.append(neigh_s[i][j],tna[i][j]).reshape(1,-1) for j in range(len(neigh_s[i]))],axis=0) for i in range(self.T)]


            target_m = rew_estimator.get_deepsets_result(comb_sa)
            pred_m = rew_estimator.get_deepsets_result(neigh_x)
            # ident_term2 = (np.sum(target_m == pred_m,axis=1) == pred_m.shape[1])
            ident_term2 = equal_m(target_m, pred_m, self.eql_thresh)

            a = self.actions[i]
            ta = self.a_tgt[i]
            ident_term1 = (a==ta)
            
            ident = ident_term1*ident_term2
#             if len(np.where(ident)[0]) != 0:
            self.idents.append(np.where(ident)[0])
                
        print('Done!')
#         print('Qualified idx is {}'.format(self.idents))
        # return self.idents
    
    def  cal_prob(self, rew_estimator, mod, rep, noise, std):
        print('Start Calculating Propensity Score...')
        self.filter(rew_estimator)
        print('idenst is  {}'.format(self.idents))
        if np.sum([len(x) for x in self.idents])>0:
            batch_idx = idx2batch(self.idents, self.N, self.T)
            s, x, r, neigh_s, neigh_x = batch_generator(batch_idx, self.trajs, self.adj_mat)
            a, tna = action_generator(batch_idx, self.a_tgt, self.adj_mat) #[[neigh_ta],...batch...,]
            comb_sa = [np.concatenate([np.append(neigh_s[i][j],tna[i][j]).reshape(1,-1) for j in range(len(neigh_s[i]))],axis=0) for i in range(len(batch_idx))]

            target_m = rew_estimator.get_deepsets_result(comb_sa)
            if mod == 'MC':
                prob_m = self.MC_proc(neigh_s, target_m, rew_estimator, self.beh_pi, rep)
            elif mod == 'TR':
                prob_m = self.traversal_proc(neigh_s, target_m, rew_estimator, self.beh_pi)
            if noise:
                prob_m = prob_m * (1+std * np.random.uniform(0,1,prob_m.shape)) 
            prob_m = np.minimum(np.maximum(prob_m, self.thresh),1)

            prob = prob_m * self.beh_pi
        else:
            prob = np.array([1])
            batch_idx = np.array([0])

        return batch_idx, prob

    def MC_proc(self, neigh_s, target_m, rew_estimator, beh_pi, rep):
        print('Now MC procedure ...')
        count = np.zeros(len(target_m))
        for i in tqdm(range(rep)):
            
            comb_sa = [np.concatenate([x,np.random.binomial(1,beh_pi,len(x)).reshape(-1,1)],axis=1) for x in neigh_s]

            pred_m = rew_estimator.get_deepsets_result(comb_sa)
            
            count += equal_m(pred_m, target_m,  self.eql_thresh).astype(int)
            # count += (np.sum(target_m == pred_m,axis=1) == pred_m.shape[1]).astype(int)
        print('Done!')
        return count/rep

    def traversal_proc(self, neigh_s, target_m, rew_estimator, beh_pi):
        print('Now traversal procedure ...')
        probs = []
        for i,x in tqdm(enumerate(neigh_s)):
            wrapped_x, postv_num = wrap_traversal(x)
            # neg_num = len(x) - postv_num
            pred_m = rew_estimator.get_deepsets_result(wrapped_x)
            target = np.repeat(target_m[i].reshape(1,-1),len(wrapped_x),axis=0)
            ident = equal_m(pred_m, target, self.eql_thresh)
            num_pos = postv_num[np.where(ident)[0]]
            # num_neg = neg_num[np.where(ident)[0]]
            prob = np.sum([np.power(beh_pi,i)*np.power(1-beh_pi,len(x)-i) for i in num_pos])
            probs.append(prob)
        print('Done!')
        return np.array(probs)




        
        
    
