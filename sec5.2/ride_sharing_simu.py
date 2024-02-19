from utils import *
from tqdm import tqdm
import math
import distribution


"""
    1. behaviour policy: always random 0.5 reward (DG-level)
    2. target policy
        1. based on generated u_O (simu-level), get fixed reward positions. (simu-level)
    
    Output:
        data: a len-N list. data[i] is a len-T list, where data[i][t] is [S_{i,t}, A_{i,t}, R_{i,t}]; 
            for now, we only use the state in data[i][T]
        adj_mat: binary adjacent matrix
        [[O, D, M], A, R]
"""
def saw_cutfunc(x):
    saw = -(x-0.5)**2 +1
    return saw


def gen_D(O, D, connect, adj_mat, change_factor):
    delta = D-O
    connect = saw_cutfunc(connect)
    connect_mat = np.minimum(adj_mat*connect, connect[:,None]) # each row being the connectivity of each grid's adjacent grids
    bi_connect = connect_mat#saw_cutfunc(connect_mat) # the connectivity between grids is defined to be the minimum of the connectivity of these grids themselves
    gap = delta[:,None] - np.tile(delta, (delta.shape[0],1)) # The D-O gap between each grid all grids
    table = bi_connect*gap # The number of drivers that should be send from each grid to each grid, ith row jth column is the number of drivers that should be send from grid i to grid j
    change  = table.sum(axis = 1) / adj_mat.sum(axis = 1)  # control the degree of change
    return D - change



def DG_once(cfg, p=0.5, a_tgt = None , u_D = None):  
    """ 
    Gen behav data
    """
    # print("Generating data...")
    l = cfg.DATA.l
    T_burn_in = cfg.DATA.burn_in
    u_O = distribution.u_O[str(l)]
    T = cfg.DATA.T + T_burn_in
    factor = cfg.DATA.factor
    connect = distribution.connectivity[str(l)]
     

    N = l ** 2
    adj_mat = getAdjGrid(l)
    adj_mat = adj_mat - np.eye(len(adj_mat))
    # initialization
     
    R = []
    if a_tgt is not None: 
        A = a_tgt
    else:
        A = rbin(1, p, (N, T+1))

    bubble_map = np.zeros((N,T+1))
    bubble_map[A==0] = 1
    bubble_map[A!=0] = 1.3


    S_base = rpoisson(u_O, (T+1,N)).T.astype(int) if cfg.TRAIN.order == "P" else rnorm(u_O, cfg.DATA.sd_R, (T+1,N)).T.astype(int)
    O = S_base*bubble_map

    D = [arr([cfg.DATA.u_D for _ in range(N)])] 
    M = [(1 - abs(D[0] - O[:, 0]) / (1 + D[0] + O[:, 0]))]
    S = [S_base[:,0]]

    """ MAIN: state transition and reward calculation [no action selection]
    """
    for t in range(1, T+1): 
        _O_t, _D_t = O[:, t-1] , D[-1]

        D_t = gen_D(_O_t, _D_t, connect, adj_mat, factor)
        S_t = S_base[:, t] # + (1-cfg.DATA.w_O) * _O_t
        # O_t = O[:, t] #cfg.DATA.w_O * O[:, t] + (1-cfg.DATA.w_O) * _O_t
        
        M_t = cfg.DATA.w_M * (1 - abs(D_t - _O_t) / (1 + D_t + _O_t)) + (1 - cfg.DATA.w_M) * M[-1]
        R_t_1 =  M_t**2 * np.minimum(D_t,_O_t) - 2 * abs(D_t - _O_t) 
        
        S.append(S_t)
        D.append(D_t)
        M.append(M_t)
        R.append(R_t_1)

    """ organization
    """
    ## organization and burn-in; N * T
    R = arr(R).T[:, T_burn_in:]
    D = arr(D).T[:, T_burn_in:T]
    M = arr(M).T[:, T_burn_in:T]
    S = arr(S).T[:, T_burn_in:T]
    A = A[:, T_burn_in:T]
    fc = np.tile(connect, (T-T_burn_in,1)).T
    num_neigh = np.tile(adj_mat.sum(axis = 1), (T-T_burn_in,1)).T
    #print(num_neigh)
    # print(R.shape, D.shape, M.shape, S.shape, A.shape, fc.shape)
    ## reorganization
    data = []
    for i in range(N):
        data_i = []
        for t in range(T - T_burn_in):
            data_i.append([arr([S[i, t], D[i, t], M[i, t], fc[i,t], num_neigh[i,t]]), A[i, t], R[i, t]])
        data.append(data_i)

    raw = [[S, D, M, fc, num_neigh], A, R]

    return data, adj_mat, raw

