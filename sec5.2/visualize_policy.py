from ride_sharing_simu import *
from multi_stage_simu import *
from utils import *
from policy import TopPolicy
import numpy as np
import argparse
import matplotlib.pyplot  as plt
import matplotlib.gridspec as gridspec
from config import Config
import pickle
from matplotlib.transforms import Bbox as Bbox
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--top', type=int, default=15) 
parser.add_argument('--deepset', type=int, default=1)
parser.add_argument("--obj", type = int, default = 0) # 0 for Order policy
parser.add_argument("--name", type = str, default= "default")
parser.add_argument("--mode", type = str, default = "DEBUG")
parser.add_argument("--l", type = int, default = 15)

args = parser.parse_args()

cfg = Config(args)
l = cfg.DATA.l
T = cfg.DATA.T
# 左上角较大，右下角较小

data, adj_mat, raww = DG_once(cfg, p = 0.5)

# Get the initial driver combine it with actions see if stable.

raws = []
TOP = [0,30,40,50]
for top in TOP:
    tgt_pi = TopPolicy(None, top, distribution.u_O)
    states = np.zeros((1, l**2, T+cfg.DATA.burn_in))
    a_tgt = tgt_pi.get_action(states)[0]
    if top == 0:
        data, adj_mat, raw = DG_once(cfg=cfg)
        raws.append(raw)
        continue
    data, adj_mat, raw = DG_once(cfg=cfg, a_tgt= a_tgt)
    raws.append(raw)

fig = plt.figure(tight_layout = True, figsize=(9,3),dpi = 400)
gs = gridspec.GridSpec(1,3)
ax = fig.add_subplot(gs[0, 0]) 
plt.plot(np.arange(T), np.mean(raws[0][0][2],axis = 0), label = "{}".format("Behav"))
plt.plot(np.arange(T), np.mean(raws[1][0][2],axis = 0), label = "TOP{}".format(TOP[1]))
plt.plot(np.arange(T), np.mean(raws[2][0][2],axis = 0), label = "TOP{}".format(TOP[2]))
plt.plot(np.arange(T), np.mean(raws[3][0][2],axis = 0), label = "TOP{}".format(TOP[3]))
plt.legend()
plt.title("Average mismatch")

ax = fig.add_subplot(gs[0, 1]) 
plt.plot(np.arange(T), np.mean(raws[0][-1],axis = 0), label = "{}".format("Behav"))
plt.plot(np.arange(T), np.mean(raws[1][-1],axis = 0), label = "TOP{}".format(TOP[1]))
plt.plot(np.arange(T), np.mean(raws[2][-1],axis = 0), label = "TOP{}".format(TOP[2]))
plt.plot(np.arange(T), np.mean(raws[3][-1],axis = 0), label = "TOP{}".format(TOP[3]))
plt.legend()
plt.title("Average reward")

ax = fig.add_subplot(gs[0, 2]) 
sns.heatmap(distribution.u_O[str(l)].reshape(l,l) , xticklabels=5,
yticklabels=5, square= True, cmap = "YlGnBu", cbar= True)
plt.title("Average number of Orders")

os.makedirs("visulize_p/", exist_ok=True)
plt.savefig("visulize_p/{}policy_l={}.png".format(cfg.CLS.policy_type, l))