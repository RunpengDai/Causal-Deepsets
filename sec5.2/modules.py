import copy
import numpy as np
import random
# from multi_stage.utils import to_tensor
from utils import *
from policy import TopPolicy
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch._C import device
from torch.types import Device
from tqdm import tqdm
from torch import nn, randint
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import distribution
import time

def wrap_neighbor_trajs(trajs, K):
    '''
    concatenate adjacent time axis state-action pair to be a new state
    aim to meet the markov assumption


    output:

    data type: [[S_(t-K+1),A_(t-K+1),S_(t-K+2),...,S_(t-1),A_(t-1),S_t], A_t, R_t]
    len(out),len(out[0]),len(out[0][0]) = R,N,T
    new S_dim  = (old S_dim + 1) * (K-1) + old S_dim 

    '''
    R,N,T = len(trajs),len(trajs[0]),len(trajs[0][0])
    data = []
    for r in range(R):
        data_r = []
        for i in range(N):
            data_i = []
            for t in range(T-K+1):
                state = np.concatenate([np.append(trajs[r][i][t+k][0],trajs[r][i][t+k][1]) for k in range(K-1)]+[trajs[r][i][t+K-1][0]])
                data_i.append([state, trajs[r][i][t+K-1][1], trajs[r][i][t+K-1][2]])
            for rep in range(K-1):
                t = 0
                state = np.concatenate([np.append(trajs[r][i][t+k][0],trajs[r][i][t+k][1]) for k in range(K-1)]+[trajs[r][i][t+K-1][0]])
                data_i.insert(0,[state, trajs[r][i][t+K-1][1], trajs[r][i][t+K-1][2]])
            data_r.append(data_i)
    
        data.append(data_r) 
    return data


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

# 其实考虑把deepset作为父类，其他从他继承来，这样可能比较方便。如果需要DIY，再单独加？

class Reward_Estimator(nn.Module):
    def __init__(self, deepset, dims, spatial):
        super(Reward_Estimator, self).__init__()
        self.deepset = deepset
        self.spatial = spatial
        x_dim, ts_dim, deep_dim = dims["inputs"]
        x_hdim, ts_hdim, deep_hdim, out_hdim= dims["hidden"]
        x_outdim, ts_outdim, deep_outdim = dims["outputs"]

        if spatial==True:
            self.dim_sum = sum(dims["outputs"]) if deepset else x_outdim*2 + ts_outdim
        else:
            self.dim_sum = x_outdim + deep_outdim if deepset else x_outdim + x_dim
                
        self.feature_extractor =nn.Sequential(
            nn.Linear(x_dim, x_hdim),
            nn.ReLU(inplace = True),
            nn.Linear(x_hdim, x_hdim*2),
            nn.ReLU(inplace = True),
            nn.Linear(x_hdim*2, x_outdim),
            nn.ReLU(inplace = True),
        )

        self.temp_spat_encoding = nn.Sequential(
            nn.Linear(ts_dim, ts_hdim),
            nn.ReLU(inplace = True),
            nn.Linear(ts_hdim, ts_outdim),
            nn.ReLU(inplace = True)
        )

        self.ds = Deepsets(deep_dim, deep_hdim, deep_outdim)
        self.mf = nn.Sequential(
                        nn.Linear(deep_dim, deep_hdim),
                        #nn.BatchNorm1d(2*out_hdim),
                        #nn.ReLU(inplace=True),
                        #nn.Dropout(0.4),
                        #nn.Linear(deep_hdim, deep_hdim),
                        nn.ReLU(inplace=True),
                        nn.Linear(deep_hdim, deep_outdim),
                        nn.ReLU(inplace=True)                         
                        )


        self.output_layer = nn.Sequential(
                        nn.Linear(self.dim_sum, out_hdim),
                        #nn.BatchNorm1d(2*out_hdim),
                        #nn.ReLU(inplace=True),
                        #nn.Dropout(0.4),
                        #nn.Linear(2*out_hdim, out_hdim),
                        nn.ReLU(inplace=True),
                        nn.Linear(out_hdim, int(out_hdim/2)),
                        nn.ReLU(inplace=True),
                        nn.Linear(int(out_hdim/2), 1)                           
                        )


# feature ->10, ->6 133 MSE
    def forward(self, x, neigh_x , temp_spat_x):
        x = self.feature_extractor(x)
#         print('temp spat x shape is ',format(temp_spat_x.shape))
        
        if self.deepset:           
            neigh = self.ds(neigh_x) # shape of neigh_x [Num of neighbors, 4(o,d,a,m)]
            #print(neigh.shape)
        else:
            neigh = torch.cat([torch.mean(x,0).view(1,-1) for x in neigh_x])
            #print(neigh.shape)
        if self.spatial == True:
            s = self.temp_spat_encoding(temp_spat_x)
            out = torch.cat([x, neigh, s],axis=1)
        else:
            out = torch.cat([x, neigh],axis=1)
        return self.output_layer(out)



class Omega_network(nn.Module):
    def __init__(self, deepset, dims, spatial):
        super(Omega_network, self).__init__()
        self.deepset = deepset
        self.spatial = spatial

        x_dim, ts_dim, deep_dim = dims["inputs"]
        x_hdim, ts_hdim, deep_hdim, out_hdim= dims["hidden"]
        x_outdim, ts_outdim, deep_outdim = dims["outputs"]

        if spatial==True:
            self.dim_sum = sum(dims["outputs"]) if deepset else x_outdim +x_dim + ts_outdim
        else:
            self.dim_sum = x_outdim + deep_outdim if deepset else x_outdim + x_dim
                
        self.feature_extractor =nn.Sequential(
                    nn.Linear(x_dim, x_hdim),
                    nn.ReLU(inplace = True),
                    nn.Linear(x_hdim, x_hdim*2),
                    nn.ReLU(inplace = True),
                    nn.Linear(x_hdim*2, x_outdim),
                    nn.ReLU(inplace = True),
                )

        self.temp_spat_encoding = nn.Sequential(
            nn.Linear(ts_dim, ts_hdim),
            nn.ReLU(inplace = True),
            nn.Linear(ts_hdim, ts_outdim),
            nn.ReLU(inplace = True)
        )
        
        self.output_layer = nn.Sequential(
                        nn.Linear(self.dim_sum, 2*out_hdim), 
                        nn.ReLU(inplace=True),
                        nn.Linear(2*out_hdim, out_hdim),
                        nn.ReLU(inplace=True),
                        nn.Linear(out_hdim, int(out_hdim/2)),
                        nn.ReLU(inplace=True),
                        nn.Linear(int(out_hdim/2), 1)                           
                        )

# feature ->10, ->6 133 MSE
    def forward(self, x, neigh , temp_spat_x):
        x = self.feature_extractor(x)
        if self.spatial == True:
            ts = self.temp_spat_encoding(temp_spat_x)
            out = torch.cat([x, neigh, ts],axis=1)
        else:
            out = torch.cat([x, neigh],axis=1)
        
        out = self.output_layer(out)
        out = torch.log(1+torch.clip(torch.exp(out),min=None, max=1e6))
        out = out/(1e-8+torch.mean(out)) #normalization
        return out


class F_network(nn.Module):
    def __init__(self, deepset, x_dim, temp_spat_dim, hidden_dim, deep_dim):
        super(F_network, self).__init__()
 
        self.deepset = deepset
        self.feature_extractor=nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, 10),

        )
        if deepset:
            self.output_layer = nn.Sequential(
                                        nn.BatchNorm1d(deep_dim + 10),
                                        nn.Linear(1 + 10, hidden_dim),
                                        nn.Linear(deep_dim + temp_spat_dim + 10, hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, 1)                           
                                        )
        else:
            self.output_layer = nn.Sequential(
                                nn.BatchNorm1d(input_dim*2),
                                nn.Linear(x_dim *2 + temp_spat_dim, hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, int(hidden_dim/2)),
                                nn.ReLU(inplace=True),
                                nn.Linear(int(hidden_dim/2), 1)                           
                                )
    def forward(self, x, neigh, temp_spat_x):
        if self.deepset:
            x = self.feature_extractor(x)
        out = torch.cat([x,neigh,temp_spat_x],axis=1)
        out = self.output_layer(out)
        out = out/(1e-8+torch.mean(out)) #normalization
        return out


class Setformer(nn.Module):
    def __init__(self,input_dim, hidden_dim, out_dim):
        super(Setformer, self).__init__()
        self.input_dim = input_dim
        self.q = nn.Linear(input_dim, hidden_dim)
        self.k = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(input_dim, hidden_dim)
        self.o = nn.Linear(hidden_dim, out_dim)
    
    def attention(self, neigh):
        q, k, v = self.q(neigh), self.k(neigh), self.v(neigh)
        scores = torch.bmm(q, k.transpose(1,2)) / math.sqrt(self.input_dim)
        #print(scores.shape)
        attention_weights = nn.Softmax(scores, axis = -1)
        #print(attention_weights.shape)
        atten_out = torch.bmm(attention_weights, v)
        return atten_out

    def forward(self, neighb):
        l = []
        for neigh in neighb:
            q, k, v = self.q(neigh), self.k(neigh), self.v(neigh)
            #print(q.shape)
            scores = torch.mm(q, k.transpose(0,1)) / math.sqrt(self.input_dim)
            #print(scores.shape)
            attention_weights = torch.softmax(scores, dim = -1)
            #print(attention_weights.shape)
            atten_out = torch.mm(attention_weights, v)
            l.append(torch.mean(self.o(atten_out),0).view(1,-1))
        return torch.cat(l)





class Deepsets(nn.Module):
    def __init__(self,input_dim, hidden_dim, out_dim):
        super(Deepsets, self).__init__()
        self.phi = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                #nn.BatchNorm1d(hidden_dim),
                                nn.ReLU(inplace = True),
                                #nn.Dropout(),
                                nn.Linear(hidden_dim,out_dim),
                                # nn.ReLU(inplace=True),
                                # nn.Linear(hidden_dim, out_dim),
                                nn.ReLU(inplace = True))
        

    def forward(self, neigh_x):
        out = torch.cat([torch.mean(self.phi(x),0).view(1,-1) for x in neigh_x])
        return out


def temp_spat_encoder(t_idx, n_idx, T, N, time_tag):
    '''
    temporal information use one-hot code
    spational information concatenate row-column corrdinates one-hot code of the point
    '''
    temp = np.eye(T)[t_idx]
    
    l = int(np.sqrt(N))
    r = n_idx // l
    c = n_idx % l
    spat = np.concatenate([np.eye(l)[r], np.eye(l)[c]],axis=1)
    if time_tag:
        temp_spat = np.concatenate([temp, spat],axis=1)
    else:
        temp_spat = spat
    
    return temp_spat

def Batch_generator(indexs, trajs, adj_mat, a_tgt, device, type = None, time_tag = False):
    '''
    returns:
    [behav_x, tgt_x, rewards, behav_neigh_x, tgt_neigh_x, temp_spat],
    [x_, neigh_x_, temp_spat_]
    '''
        
    R, N, T, batch_size , batch_idx= indexs.R, indexs.N, indexs.T, indexs.batch_size, indexs.batch_idx
    r_idx, n_idx, t_idx = indexs.r_idx, indexs.n_idx, indexs.t_idx
    
    states = np.concatenate(
        [trajs[r][i][j][0].reshape(1,-1) 
        for r,i,j in zip(r_idx,n_idx,t_idx)],axis=0)
    actions = np.concatenate(
        [trajs[r][i][j][1].reshape(1,-1) 
        for r,i,j in zip(r_idx,n_idx,t_idx)],axis=0)
    behav_x = np.concatenate([states, actions], axis=1)

    tgt_actions, tgt_neigh_a = action_generator(indexs, a_tgt, adj_mat) 
    tgt_x = np.concatenate([states, tgt_actions],axis=1)

    rewards = np.concatenate(
        [trajs[r][i][j][2].reshape(1,-1) 
        for r,i,j in zip(r_idx,n_idx,t_idx)],axis=0)
    
    neigh_s = [np.concatenate(
        [trajs[r][i][t][0].reshape(1,-1) 
        for i in np.where(adj_mat[n]==1)[0]],axis=0) 
        for r,n,t in zip(r_idx,n_idx,t_idx)]

    behav_neigh_x = [np.concatenate(
        [np.append(trajs[r][i][t][0],trajs[r][i][t][1]).reshape(1,-1) 
        for i in np.where(adj_mat[n]==1)[0]],axis=0) 
        for r,n,t in zip(r_idx,n_idx,t_idx)]

    tgt_neigh_x = [np.concatenate([np.append(neigh_s[i][j], tgt_neigh_a[i][j]).reshape(1,-1) 
        for j in range(len(neigh_s[i]))],axis=0) for i in range(indexs.batch_size)]

    
    temp_spat = temp_spat_encoder(t_idx, n_idx, T, N, time_tag)
    
    Batch =[behav_x, tgt_x, rewards, behav_neigh_x, tgt_neigh_x, temp_spat] #if RKHS== False else [behav_x, tgt_x, rewards, behav_neigh_x, tgt_neigh_x, temp_spat]

    #### Above generate current time step.

    next_indexs = Batch_idx(R, N, T, batch_size, Time=None, batch_idx = batch_idx+1)
        
    states_ = np.concatenate(
        [trajs[r][i][j][0].reshape(1,-1) for r,i,j in zip(next_indexs.r_idx, next_indexs.n_idx, next_indexs.t_idx)],axis=0) 
    
    actions_, neigh_a_ = action_generator(next_indexs, a_tgt, adj_mat) 
    x_ = np.concatenate([states_,actions_],axis=1)

    neigh_s_ = [np.concatenate(
        [trajs[r][i][t][0].reshape(1,-1) 
        for i in np.where(adj_mat[n]==1)[0]],axis=0) 
        for r,n,t in zip(next_indexs.r_idx, next_indexs.n_idx, next_indexs.t_idx)]
    
    neigh_x_ = [np.concatenate([np.append(neigh_s_[i][j],neigh_a_[i][j]).reshape(1,-1) 
                for j in range(len(neigh_s_[i]))],axis=0) for i in range(next_indexs.batch_size)]

    temp_spat_ = temp_spat_encoder(next_indexs.t_idx, next_indexs.n_idx, T, N, time_tag)

    Batch_ = [x_, neigh_x_, temp_spat_]
    #### Above generate next time step

    if type is not None:
        
        Batch, Batch_ = [to_tensor(x, device) for x in Batch], [to_tensor(x, device) for x in Batch_]

    return Batch, Batch_


def action_generator(indexs, tgt_action, adj_mat):
    R, N, T = indexs.R, indexs.N, indexs.T
    r_idx, n_idx, t_idx = indexs.r_idx, indexs.n_idx, indexs.t_idx
    # print(r_idx)
    # print(n_idx)
    # print(t_idx)
    # print(tgt_action)
    a = np.concatenate(
        [tgt_action[r][i][j].reshape(1,-1) for r,i,j in zip(r_idx,n_idx,t_idx)],axis=0)

    ta = [np.concatenate([tgt_action[r][i][t].reshape(-1,1) for i in np.where(adj_mat[n]==1)[0]],axis=0) for r,n,t in zip(r_idx,n_idx,t_idx)]
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
    action_matrix = np.concatenate([np.array(list(np.binary_repr(i,bit))).astype(np.int64).reshape(1,-1) for i in range(2**bit)],axis=0)
    neigh = [np.concatenate([x,act.reshape(-1,1)],axis=-1) for act in action_matrix]
    return neigh, np.sum(action_matrix, axis=1)




class FQE_module():
    def __init__(self, trajs, adj_mat, u0s, cfg):
        
        self.deepset = cfg.CLS.deepset
        self.model = Reward_Estimator(cfg.CLS.deepset, cfg.TRAIN.dims, cfg.TRAIN.spatial).to(cfg.CLS.device)
        self.target_model = Reward_Estimator(cfg.CLS.deepset, cfg.TRAIN.dims, cfg.TRAIN.spatial).to(cfg.CLS.device)
        self.update_target()
        
        
        
        self.trajs = trajs
        self.adj_mat = adj_mat
        self.R = len(self.trajs)
        self.N = len(self.trajs[0])
        self.T = len(self.trajs[0][0])
        self.NTR = self.T*self.N*self.R
        self.NT = self.N * self.T
        self.gamma = cfg.CLS.gamma
        self.time_tag = cfg.CLS.time_tag
        self.stag = cfg.CLS.stag
        
        self.cfg = cfg
        self.writer = SummaryWriter(self.cfg.CLS.log_pth) if cfg.mode == "DEBUG" else None
        
        tgt_pi = TopPolicy(None, cfg.CLS.topk, distribution.u_O)
        states = np.zeros((self.R, self.N, self.T))
        self.a_tgt = tgt_pi.get_action(states)

        
        self.device = cfg.CLS.device
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = cfg.TRAIN.qlr)
        # self.scheduler = MultiStepLR(self.optimizer, milestones=cfg.TRAIN.cropping, gamma=cfg.TRAIN.gamma)
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def train(self):
        
        print('Start Training Reward Function Procedure...')
        self.model.train()


        batch_size =self.cfg.TRAIN.batch_size
        gross_train_iters = self.cfg.TRAIN.gross_train_iters
        target_freq = self.cfg.TRAIN.target_freq

        bar = tqdm(range(gross_train_iters), mininterval= 50)
        for i in bar:  
            if (i+1) % target_freq ==0:
                self.update_target()

            if self.stag == -1:
                indexs = Batch_idx(self.R,self.N,self.T,batch_size)
            else:
                batch_idx = np.random.choice(self.R, size=batch_size)*self.T*self.N +self.T*self.stag+np.random.choice(self.T-1, size = batch_size)
                indexs = Batch_idx(self.R, self.N, self.T, batch_size, Time = None, batch_idx=batch_idx)
            
            loss = self.train_one_step(indexs, i)
            if self.cfg.mode =="DEBUG":
                self.writer.add_scalar("train", loss, i)
            
            if (i+1) == 10000:
                batch_idx = np.arange(0,self.N*self.T*self.R,self.T)
                indexs = Batch_idx(self.R, self.N, self.T, len(batch_idx), Time=None, batch_idx = batch_idx)
                Batch, _ = Batch_generator(indexs, self.trajs, self.adj_mat, self.a_tgt, self.device, "tensor",self.time_tag)
                x, _, r, neigh_x, _, ts = Batch
                q = self.get_value(x, neigh_x, ts).reshape(indexs.batch_size, -1)
                if int(loss) >= 1e5:
                    return False
                if int(torch.sum(q/self.R/self.N*(1-self.gamma))) < 30:
                    return False

            if (i+1) % self.cfg.test_frequency ==0 and self.cfg.mode =="DEBUG":
                batch_idx = np.arange(0,self.N*self.T*self.R,self.T)
                indexs = Batch_idx(self.R, self.N, self.T, len(batch_idx), Time=None, batch_idx = batch_idx)
                Batch, _ = Batch_generator(indexs, self.trajs, self.adj_mat, self.a_tgt, self.device, "tensor",self.time_tag)
                x, _, r, neigh_x, _, ts = Batch
                q = self.get_value(x, neigh_x, ts).reshape(indexs.batch_size, -1)
                self.writer.add_scalar("test", torch.sum(q/self.R/self.N*(1-self.gamma)), i)
        return True

           

            
    def train_one_step(self, indexs, i):
    
        Batch_t, Batch_t_ = Batch_generator(indexs, self.trajs, self.adj_mat, self.a_tgt, self.device, "tensor",self.time_tag)  
        x, _, r, neigh_x, _, ts = Batch_t
        x_, neigh_x_, ts_ = Batch_t_
        dones = torch.unsqueeze(to_tensor(indexs.dones, self.device),1)
        
        with torch.no_grad():
            self.target_model.eval()
            if self.time_tag==False:
                tgt = r + self.gamma * self.target_model(x_,neigh_x_,ts_).detach()
            else:
                tgt = r + self.gamma * (1-dones)* self.target_model(x_,neigh_x_,ts_).detach()

        self.optimizer.zero_grad()
        pred = self.model(x, neigh_x, ts)
        # if self.cfg.mode =="DEBUG":
        #     self.writer.add_scalar("TD-error", torch.mean(tgt-pred), i)
        loss = self.loss_fn(pred, tgt)
        loss.backward()
        self.optimizer.step()
        #self.scheduler.step()

        return loss.detach().cpu().numpy()

    def get_value(self, x, neigh_x, ts):
        with torch.no_grad():
            self.model.eval()
            pred = self.model(x,neigh_x,ts)
        self.model.train()
        return pred

    def get_deepsets_result(self, neigh_x):
        with torch.no_grad():
            self.model.eval()
            if self.deepset:
                pred = self.model.ds(neigh_x)
            else:
                pred = torch.cat([torch.mean(x,0).view(1,-1) for x in neigh_x])
            self.model.train()
        return pred.detach().cpu().numpy()
        

class Density_Ratio_RKHS():
    def __init__(self, trajs, adj_mat, u0s, value_func, cfg):
        
        self.cfg = cfg

        self.trajs = trajs
        # self.normalize = normalize
        # if self.normalize:
        #     self.trajs, self.min_max = self.preprocess(self.trajs_ori)  # self.miin_max is a list containing
        #                                                                 # suprema of states and rewards
        # else:
        #     self.trajs = self.trajs_ori
        
        self.adj_mat = adj_mat
        self.R = len(self.trajs)
        self.N = len(self.trajs[0])
        self.T = len(self.trajs[0][0])
        self.NTR = self.T*self.N*self.R
        self.NT = self.N * self.T
        self.gamma = cfg.CLS.gamma

        tgt_pi = TopPolicy(self.trajs, cfg.CLS.topk, u0s)
        states = np.zeros((self.R, self.N, self.T))
        self.a_tgt = tgt_pi.get_action(states)
        self.time_tag = cfg.CLS.time_tag
        self.stag = cfg.CLS.stag
        # print(len(self.a_tgt))

        self.value_func = value_func
        self.value_func.model.eval()
        self.w = Omega_network(cfg.CLS.deepset, cfg.TRAIN.dims, cfg.TRAIN.spatial).to(cfg.CLS.device)
        
        self.device = cfg.CLS.device
        self.loss_fn = nn.MSELoss()
        self.optimizerW = optim.Adam(self.w.parameters(), lr = cfg.TRAIN.wlr)
        self.med_dist = self.get_med_dist()
        
        # self.writer = SummaryWriter(self.cfg.CLS.log_pth) 

    
    def get_med_dist(self):
        '''
        use all samples to estimate the median of difference between each combination
        '''
        
        idx = Batch_idx(self.R, self.N, self.T, batch_size=int(self.NTR/1000))
        Batch_t, _ = Batch_generator(idx, self.trajs, self.adj_mat, self.a_tgt, self.device,'tensor',self.time_tag)  
        x, _, _, neigh_x, _, ts = Batch_t
        neigh = self.value_func.get_deepsets_result(neigh_x)
        #ts = self.value_func.model.spat_encoding(ts).detach().numpy()

        s_a = np.concatenate([to_numpy(x),neigh], axis=1)
        med_dist = np.median(np.sqrt(np.sum(np.square(s_a[None, :, :] - s_a[:, None, :]), axis = -1)))
        return med_dist
    
    def K(self, x, x_):
        '''
        Gaussian kernel
        input: x:(batch_size1, dim), x_:(batch_size2, dim)
        output: kernel matrix :(batch_size1, batch_size2)
        '''
        x = torch.cat(x,1)
        x_ = torch.cat(x_,1)
        diff = torch.unsqueeze(x, 1) - torch.unsqueeze(x_,0)
        k = torch.exp(-torch.sum(torch.square(diff), axis = -1)/(2.0*self.med_dist*self.med_dist))
        return k

        
    def train(self):
        print('Start RKHS density ratio Procedure...')

        batch_size = self.cfg.TRAIN.d_batch_size
        max_iters = self.cfg.TRAIN.density_iters
        print_freq = self.cfg.TRAIN.print_freq        

        bar = tqdm(range(max_iters), mininterval= 200)
        for i in bar:
            if self.stag == -1:
                
                idx_g = Batch_idx(self.R, self.N, self.T, batch_size) 
                idx_0 = Batch_idx(self.R, self.N, self.T, batch_size, [0])
                _idx_g = Batch_idx(self.R, self.N, self.T, batch_size) 
                _idx_0 = Batch_idx(self.R, self.N, self.T, batch_size, [0])            
            else:
                batch_idx = np.random.choice(self.R, size=batch_size)*self.T*self.N +self.T*self.stag+np.random.choice(self.T-1, size = batch_size)
                idx_g = Batch_idx(self.R, self.N, self.T, batch_size,Time=None,batch_idx=batch_idx) 
                batch_idx = np.random.choice(self.R, size=batch_size)*self.T*self.N +self.T*self.stag
                idx_0 = Batch_idx(self.R, self.N, self.T, batch_size,Time=None,batch_idx=batch_idx) 
        
                batch_idx = np.random.choice(self.R, size=batch_size)*self.T*self.N +self.T*self.stag+np.random.choice(self.T-1, size = batch_size)
                _idx_g = Batch_idx(self.R, self.N, self.T, batch_size,Time=None,batch_idx=batch_idx) 
                batch_idx = np.random.choice(self.R, size=batch_size)*self.T*self.N +self.T*self.stag
                _idx_0 = Batch_idx(self.R, self.N, self.T, batch_size,Time=None,batch_idx=batch_idx) 
            
            lossw = self.train_one_step(idx_g,_idx_g,idx_0,_idx_0)


                
    def train_one_step(self, idx_g, _idx_g, idx_0, _idx_0):
        '''
        x_ means x with time t+1
        _x_ means another sample of x with time t+1
        '''
        Batch_0, _ = Batch_generator(idx_0, self.trajs, self.adj_mat, self.a_tgt, self.device, "tensor",self.time_tag)  
        _, x0, _, _, neigh_x0, ts0 = Batch_0
        
        Batch_t, Batch_t_ = Batch_generator(idx_g, self.trajs, self.adj_mat, self.a_tgt, self.device, "tensor", self.time_tag)  
        x, _, _, neigh_x, _, ts = Batch_t
        x_, neigh_x_, ts_ = Batch_t_
        # s in the paper

        _Batch_0, _ = Batch_generator(_idx_0, self.trajs, self.adj_mat, self.a_tgt, self.device, "tensor",self.time_tag)  
        _, _x0, _, _, _neigh_x0, _ts0 = _Batch_0
        
        _Batch_t, _Batch_t_ = Batch_generator(_idx_g, self.trajs, self.adj_mat, self.a_tgt, self.device, "tensor",self.time_tag)  
        _x, _, _, _neigh_x, _, _ts = _Batch_t
        _x_, _neigh_x_, _ts_ = _Batch_t_
        # s_bar in the paper, independant sample of s
        
        # use deepset in trained deepset model in FQE module
        neigh0 = to_tensor(self.value_func.get_deepsets_result(neigh_x0), self.device)
        neigh = to_tensor(self.value_func.get_deepsets_result(neigh_x), self.device)
        neigh_ = to_tensor(self.value_func.get_deepsets_result(neigh_x_), self.device)

        _neigh0 = to_tensor(self.value_func.get_deepsets_result(_neigh_x0), self.device)
        _neigh = to_tensor(self.value_func.get_deepsets_result(_neigh_x), self.device)
        _neigh_ = to_tensor(self.value_func.get_deepsets_result(_neigh_x_), self.device)

        self.w.zero_grad()
        s_a, s_a_, s_a0 = (x, neigh, ts), (x_, neigh_, ts_), (x0, neigh0, ts0)
        _s_a, _s_a_, _s_a0 = (_x, _neigh, _ts), (_x_, _neigh_, _ts_), (_x0, _neigh0, _ts0)

        omega = torch.squeeze(self.w( *s_a ))
        _omega = torch.squeeze(self.w( *_s_a ))

        t1 = torch.mean(self.gamma**2 * torch.unsqueeze(omega,1) * self.K(s_a_, _s_a_)* torch.unsqueeze(_omega ,0))
        t2 = -2 * self.gamma * torch.mean(torch.unsqueeze(omega,1)  * self.K(s_a_, _s_a) * torch.unsqueeze(_omega,0))
        t3 = self.gamma * (1-self.gamma) * torch.mean(torch.unsqueeze(omega,1) * self.K(s_a_, _s_a0))
        t5 = torch.mean(torch.unsqueeze(omega,1) * self.K(s_a, _s_a) * torch.unsqueeze(_omega,0))
        t6 = -(1-self.gamma) * torch.mean(torch.unsqueeze(omega,1)  * self.K(s_a, _s_a0))
        t7 = self.gamma * (1-self.gamma) * torch.mean(torch.unsqueeze(_omega,1) * self.K(_s_a_, s_a0))
        t8 = -(1-self.gamma) * torch.mean(torch.unsqueeze(_omega,1)  * self.K(_s_a, s_a0))
        
        
        nuisance = (1-self.gamma)**2 * torch.mean(self.K(s_a0,_s_a0))
        # loss = torch.square(t1+ t2 + t3 + t5 + t6 + t7 + t8 + nuisance)
        loss = t1+ t2 + t3 + t5 + t6 + t7 + t8 + nuisance
        loss.backward()
        self.optimizerW.step()
        
        return loss.detach().cpu().numpy()

    def get_omega_value(self, x, neigh_x, ts):
        
        neigh = to_tensor(self.value_func.get_deepsets_result(neigh_x), self.device)
        pred = self.w(x,neigh,ts)
        return pred
   
class Density_Ratio_GAN():
    def __init__(self, gamma, trajs, adj_mat, u0s, value_func,
                 cfg):
        
        self.cfg = cfg

        self.trajs = trajs
        self.adj_mat = adj_mat
        self.R = len(self.trajs)
        self.N = len(self.trajs[0])
        self.T = len(self.trajs[0][0])
        self.NTR = self.T*self.N*self.R
        self.NT = self.N * self.T
        self.gamma = cfg.CLS.gamma
        self.time_tag = cfg.CLS.time_tag
        self.stag = cfg.CLS.stag
        
        self.value_func = value_func

        tgt_pi = TopPolicy(self.trajs, cfg.CLS.topk, u0s)
        states = np.zeros((self.R, self.N, self.T))
        self.a_tgt = tgt_pi.get_action(states)
        
        self.f = F_network(cfg.CLS.deepset, cfg.TRAIN.input_dim, cfg.TRAIN.ts_dim, cfg.TRAIN.hidden_dim, cfg.TRAIN.deep_dim)
        self.w = Omega_network(cfg.CLS.deepset,cfg.TRAIN.dims, cfg.TRAIN.spatial)
        
        self.device = cfg.CLS.device
        self.loss_fn = nn.MSELoss()
        self.optimizerF = optim.Adam(self.f.parameters(), lr = cfg.TRAIN.wlr)
        self.optimizerW = optim.Adam(self.w.parameters(), lr = cfg.TRAIN.wlr)
        
        self.writer = SummaryWriter(self.cfg.CLS.log_pth) 
        
        
    def train(self):
        print('Start GAN density ratio Procedure...')
        self.f.train()
        self.w.train()

        batch_size = self.cfg.CLS.d_batch_size
        max_iters = self.cfg.CLS.density_iters
        print_freq = self.cfg.CLS.print_freq
        bar = tqdm(range(max_iters))
        for i in bar:
            if self.stag == -1:
                idx_g = Batch_idx(self.R, self.N, self.T, batch_size) 
                idx_0 = Batch_idx(self.R, self.N, self.T, batch_size, [0])
            else:
                batch_idx = np.random.choice(self.R, size=batch_size)*self.T*self.N +self.T*self.stag+np.random.choice(self.T-1, size = batch_size)
                idx_g = Batch_idx(self.R, self.N, self.T, batch_size,Time=None,batch_idx=batch_idx) 
                batch_idx = np.random.choice(self.R, size=batch_size)*self.T*self.N +self.T*self.stag
                idx_0 = Batch_idx(self.R, self.N, self.T, batch_size,Time=None,batch_idx=batch_idx) 
   
            lossf, lossw = self.train_one_step(idx_g,idx_0)
            self.writer.add_scalar("train f function", lossf, i)
            self.writer.add_scalar("train density", lossw, i)
  


            if i%print_freq ==0:
                bar.set_description('MSEf :{}, MSEd:{}'.format(lossf,lossw))

                
    def train_one_step(self, idx_g, idx_0):

        Batch_0, _ = Batch_generator(idx_0, self.trajs, self.adj_mat, self.a_tgt, self.device, "tensor", self.time_tag)  
        _, x0, _, _, neigh_x0, ts0 = Batch_0
        
        Batch_t, Batch_t_ = Batch_generator(idx_g, self.trajs, self.adj_mat, self.a_tgt, self.device, "tensor",self.time_tag)  
        x, _, _, neigh_x, _, ts = Batch_t
        x_, neigh_x_, ts_ = Batch_t_
        
        neigh_ = to_tensor(self.value_func.get_deepsets_result(neigh_x_),self.device)
        neigh = to_tensor(self.value_func.get_deepsets_result(neigh_x), self.device)
        neigh0 = to_tensor(self.value_func.get_deepsets_result(neigh_x0), self.device)
        # print(neigh.shape)
        
        self.f.zero_grad()
        with torch.no_grad():
            self.w.eval() # 不激活relu,batch norm.
            omega = self.w(x, neigh, ts).detach() #(batch_size, )
            # tgt = r + self.gamma * (1-dones)* self.model(x_,neigh_x_,ts_).detach()
            # print(tgt)
            self.w.train()

        first_term = self.gamma * self.f(x_, neigh_,ts_) * omega
        second_term = self.f(x, neigh, ts) * omega
        third_term = torch.mean(self.f(x0, neigh, ts0) * (1- self.gamma))
        lossf = -1* torch.square(torch.mean(first_term - second_term) + third_term)
        lossf.backward()
        self.optimizerF.step()
        
        self.w.zero_grad()
        with torch.no_grad():
            self.f.eval()
            f1 = self.f(x_, neigh_, ts_).detach()
            f2 = self.f(x, neigh, ts).detach()
            f3 = self.f(x0, neigh0, ts0).detach()
            self.f.train()
        omega = self.w(x, neigh, ts)
        loss = torch.square(torch.mean(omega * (self.gamma*f1 - f2)) + torch.mean((1-self.gamma)*f3))
        loss.backward()
        self.optimizerW.step()
        
        return lossf.detach().cpu().numpy(),loss.detach().cpu().numpy()

    def get_omega_value(self, x, neigh_x, ts):
        neigh = to_tensor(self.value_func.get_deepsets_result(neigh_x), self.device)
        pred = self.w(x,neigh,ts)
        return pred
    

class Batch_idx():
    '''
    The correctness of this function can be checked in the debug.ipynb.
    '''
    def __init__(self, R, N, T, batch_size, Time= None, batch_idx= None):
        self.R = R
        self.N = N
        self.T = T
        self.Time = Time
        self.batch_size = batch_size
        if Time is None:
            self.batch_idx = np.random.choice(R*N, size=batch_size)*T +np.random.choice(self.T-1, size = batch_size)
        else:
            self.batch_idx = np.random.choice(R*N, size=batch_size)*T + np.random.choice(Time, size = batch_size)
        

        if batch_idx is not None:
            self.batch_idx = np.array(batch_idx)%(self.R*self.T*self.N)

        self.r_idx = self.batch_idx//(N*T)
        nt_index = self.batch_idx - self.r_idx*N*T
        self.n_idx = nt_index // T
        self.t_idx = nt_index -self.n_idx*T

        self.dones = (self.t_idx == T-1 )

