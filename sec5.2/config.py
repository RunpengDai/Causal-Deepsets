import os.path as osp
from easydict import EasyDict as edict
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
class Config():
    def __init__(self, args):
        self.real_rep = 10
        self.test_frequency = 1500
        self.lag = 1
        self.mode = args.mode

        DATA = edict()
        DATA.data_rep = 200 # at least take 2 !!! (for the sake of dataset spliting)
        DATA.T = 40
        DATA.l = args.l
        DATA.N = DATA.l**2
        DATA.version = 'old'
        DATA.factor = 0.1
        DATA.w_A = 2
        DATA.w_O = 0.5
        DATA.burn_in = 0
        DATA.u_D = 130
        DATA.sd_R = 1
        DATA.sd_O = 4
        DATA.w_M = 0.9
        DATA.w_D = 1

        TRAIN = edict()
        
        TRAIN.target_freq = 1000 #500 & 2000
        TRAIN.spatial = False   
        TRAIN.feature_num = 6
        TRAIN.order = 'N'
        TRAIN.max_iters = int(100004)
        TRAIN.gross_train_iters = int(160000)
        TRAIN.cropping =[80000, 120000, 160000]
        TRAIN.gamma = 0.2
        
        TRAIN.print_freq = 50
        TRAIN.batch_size = 64
        TRAIN.qlr = 3e-4
        TRAIN.wlr = 5e-5
        TRAIN.density_iters = int(1000)
        TRAIN.d_batch_size = 32
        


        CLS = edict()
        CLS.encode = "T"
        CLS.REPLICATES = 15
        CLS.gamma = 0.9
        CLS.mod = 'RKHS'
        CLS.beh_pi = 0.5
        CLS.data_split = False
        CLS.deepset = bool(args.deepset)
        CLS.normalize = False
        CLS.device = "cpu"
        CLS.time_tag = False # infinite stage or finite stage
        
        if CLS.time_tag:
            TRAIN.ts_dim = DATA.T + int(DATA.l*2)
        else:
            TRAIN.ts_dim = int(DATA.l*2)
        
        # x_dim, ts_dim, deep_dim 
        TRAIN.dims = {"inputs":[TRAIN.feature_num, TRAIN.ts_dim, TRAIN.feature_num], 
        "outputs":[8, 8, 8], "hidden":[6, 6, 8, 8]}
        
        CLS.stag = -1      # represent specific grid number to be treated, 
                                # -1 respresent integrating all grid together
        CLS.topk = args.top
        CLS.policy_type = "Driver" if args.obj else "Order"

        CLS.data_pth = 'data/{}feature_{}data.pkl'.format(CLS.policy_type,TRAIN.feature_num)

        CLS.log_pth = 'logs/{}/top{}/{}_Q{}k_O{}k/{}'.format(
        CLS.policy_type, CLS.topk, 
        CLS.deepset, 
        TRAIN.gross_train_iters/1000, TRAIN.density_iters/1000,
        args.name)
       

        CLS.f_pth = 'results/l{}_top{}deepset_{}.pkl'.format(
        DATA.l, CLS.topk,CLS.deepset)
        os.makedirs("results/", exist_ok=True)


        CLS.q_model_pth = 'model/pi{}_split{}_{}_T{}_m{}_N{}_stag{}_ttag_{}gamma_{}normalize_{}deepset_{}_lr{}_iter{}_{}_{}_policy.pkl'.format(
        CLS.topk,CLS.data_split,DATA.version, DATA.T,DATA.N,DATA.data_rep,
        CLS.stag,CLS.time_tag,CLS.gamma,CLS.normalize, CLS.deepset, TRAIN.qlr,TRAIN.gross_train_iters, args.name, CLS.policy_type)

        CLS.density_model_pth = 'model/pi{}_split{}_{}_T{}_m{}_N{}_{}_stag{}_ttag{}_gamma_{}normalize_{}deepset_{}_qlr{}_wlr{}_iter{}_{}_{}policy'.format(
            CLS.topk,CLS.data_split,DATA.version, DATA.T,DATA.N,DATA.data_rep,CLS.mod,
            CLS.stag,CLS.time_tag,CLS.gamma,CLS.normalize, CLS.deepset,
            TRAIN.qlr,TRAIN.wlr,TRAIN.density_iters,CLS.policy_type,
            args.name)


        self.DATA, self.TRAIN, self.CLS = DATA, TRAIN, CLS

