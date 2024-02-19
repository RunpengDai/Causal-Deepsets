import pickle
import argparse
from config import Config
import distribution
from modules import wrap_neighbor_trajs
from multi_stage_simu import DeepDR
from ride_sharing_simu import *
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import subprocess
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--top', type=int, default = 15) 
parser.add_argument('--deepset', type=int, default = 0)
parser.add_argument("--obj", type = int, default = 0) # 0 for Order policy
parser.add_argument("--name", type = str, default = "PIE")
parser.add_argument("--mode", type = str, default = "FORMAL")
parser.add_argument("--l", type = int, default = 5)
args = parser.parse_args()

cfg = Config(args)

if cfg.mode != "DEBUG":
    writer = SummaryWriter(args.name) 
config = {**cfg.DATA, **cfg.TRAIN, **cfg.CLS}
REPLICATES = 1 if cfg.mode == "DEBUG" else cfg.CLS.REPLICATES

def loading_behav_dat():
    print('loding_data {}'.format(cfg.CLS.data_pth))
    f = open(cfg.CLS.data_pth,'rb')
    meta_data = pickle.load(f)
    f.close()
    return meta_data

def gen_behav_dat():
    print("Generating behav data:")
    data,raw_d = [], []
    u0s = [distribution.u_O] * cfg.DATA.data_rep
    for _ in range(cfg.DATA.data_rep):
        d, adj_mat, raw_data = DG_once(cfg ,p=0.5)
        data.append(d)
        raw_d.append(raw_data)
    return data, raw_d, u0s, adj_mat

# 1.dr class
# 2.dr methods: est_reward, est_density_ratio
# 3.real_state_value

def estimating_realvalue(a_tgt):
    mean_values =[]
    for _ in range(cfg.real_rep):
        _, _, raw_data = DG_once(cfg, a_tgt= a_tgt)
        gammas = arr([cfg.CLS.gamma**i for i in range(cfg.DATA.T)])
        mean_values.append(np.sum(np.mean(raw_data[-1],axis =0)*gammas) * (1- cfg.CLS.gamma))

    return np.mean(mean_values), np.std(mean_values)


DR, IS, PLG = [],[],[]
STAT_VALUE = []
counter = 0


while len(PLG) <= REPLICATES:
    print('Now replicate {}'.format(counter+1))

    # Run the nvidia-smi command
    if cfg.CLS.device == "cuda":
        output = subprocess.check_output(["nvidia-smi"], text=True)
        print(output)

    data, raw_d, u0s, adj_mat = gen_behav_dat()
    if cfg.lag >1:
        data = wrap_neighbor_trajs(data, cfg.lag)
    

    dr =DeepDR(42, data, adj_mat, cfg)
    state_value, state_std = estimating_realvalue(dr.a_tgt[0])
    print('real_value is {:.1f},while standard error is {:.2f}'.format(state_value,state_std))

    ########################################
    status = dr.est_reward()
    if status == False: 
        print("This rep failed")
        continue
        
    counter += 1
    dr.est_density_ratio()

    dr_est, is_est, plg_est= dr.construct_est()
    
    if cfg.mode !="DEBUG":
        writer.add_scalar("PLG", plg_est, counter)
        writer.add_scalar("DR", dr_est, counter)
        writer.add_scalar("IS", is_est, counter)

    DR.append(to_numpy(dr_est))
    IS.append(to_numpy(is_est))
    PLG.append(to_numpy(plg_est))
    STAT_VALUE.append(state_value)
    ########################################

    print('real_value is {:.1f},while standard error is {:.2f}'.format(state_value,state_std))
    print('mean real_value is {:.1f}'.format(np.mean(STAT_VALUE)))  
    print('mean DR is {}'.format(np.mean(DR)))
    print('mean IS is {}'.format(np.mean(IS)))
    print('mean PLG is {:.1f}'.format(np.mean(PLG)))
    
    dump = [args.top, mse(DR, STAT_VALUE), mse(PLG, STAT_VALUE), ci(DR, STAT_VALUE), ci(PLG, STAT_VALUE), counter]
    print("MSE of DR is {}".format(dump[1]))
    print("MSE of PLG is {:.1f}".format(dump[2]))
    print("CI of DR is {}".format(dump[3]))
    print("CI of PLG is {}".format(dump[4]))

    ########################################
    with open(cfg.CLS.f_pth, 'wb') as f:
        pickle.dump(dump,f)
    draw_results()

