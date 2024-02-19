import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.gridspec as gridspec
from matplotlib.transforms import Bbox as Bbox
def load_data(fpth):
    f = open(fpth,'rb')
    b = pickle.load(f)
    f.close()
    res = np.array(b[1])
    res[np.isnan(res)] = -1
    print(fpth)
    print(res.shape)
    return res

def get_plot_data1(cfg):
    c = cfg[0]
    fname = 'results/new_car_l5_parm{}_th0.5_fmod{}_bmod{}_normalize_Falsedeepset_{}hdim16_ddim2.pkl'.format(
        top[0], c[0],c[1],deepset[0])

    ldata = load_data(fname)

    result = np.zeros((len(top),len(cfg),len(deepset),4,ldata.shape[1]))

    for t1,t in enumerate(top):
        for c1, c in enumerate(cfg):
            for d1,d in enumerate(deepset):

                fname = 'results/new_car_l5_parm{}_th0.5_fmod{}_bmod{}_normalize_Falsedeepset_{}hdim16_ddim2.pkl'.format(
                        t, c[0],c[1],d)
                ldata = load_data(fname)
                # print(ldata)
                result[t1,c1,d1] = ldata
    print('result is {}'.format(np.mean(result,axis=-1)))
    mse = np.mean(np.square(result[:,:,:,:3] - result[:,:,:,3][:,:,:,None,:]),axis=-1)
    error = np.std(np.square(result[:,:,:,:3] - result[:,:,:,3][:,:,:,None,:]),axis=-1)
    # print(np.mean(result[:,:,:,:,3],axis=(1,2,3,4)))
    # print(result[:,:,:,:,3])
    return mse, np.mean(result[:,:,:,3],axis=(1,2,3)),error

def get_plot_data(cfg):
    c = cfg[0]
    fname = 'results/new_car_l10_parm{}_th0.5_fmod{}_bmod{}_normalize_Falsedeepset_{}hdim16_ddim2.pkl'.format(
        top[0], c[0],c[1],deepset[0])

    ldata = load_data(fname)

    result = np.zeros((len(top),len(cfg),len(deepset),4,ldata.shape[1]))

    for t1,t in enumerate(top):
        for c1, c in enumerate(cfg):
            for d1,d in enumerate(deepset):


                fname = 'results/new_car_l10_parm{}_th0.5_fmod{}_bmod{}_normalize_Falsedeepset_{}hdim16_ddim2.pkl'.format(
                        t, c[0],c[1],d)
                ldata = load_data(fname)
                # print(ldata)
                result[t1,c1,d1] = ldata
    print('result is {}'.format(np.mean(result,axis=-1)))
    mse = np.mean(np.square(result[:,:,:,:3] - result[:,:,:,3][:,:,:,None,:]),axis=-1)
    error = np.std(np.square(result[:,:,:,:3] - result[:,:,:,3][:,:,:,None,:]),axis=-1)
#     print('mse shape {}'.format(np.square(result[:,:,:,:3] - result[:,:,:,3][:,:,:,None,:]).shape))
    # print(np.mean(result[:,:,:,:,3],axis=(1,2,3,4)))
    # print(result[:,:,:,:,3])
    return mse, np.mean(result[:,:,:,3],axis=(1,2,3)),error




top = [0.2,0.4,0.6,0.8]
deepset = [False,True]


cfg = [[1,1],[0,1],[0,0]]
result1, real_value1,std1 = get_plot_data1(cfg)

cfg = [[1,1],[0,1],[0,0]]
result2, real_value2,std2 = get_plot_data(cfg)





rdata = [result1,result2]
stdata = [std1,std2]

# cfg = [[True,'rush',4,15]]
# result1,real_value1 = get_plot_data(cfg,policy)

# result = np.concatenate([result,result1],axis=1)
# print('result shape is ')
# print(result.shape)
# cfg = [[False,'hour_sonehot',4,1],[False,'rush_sonehot',4,1],[False, 'hour_sonehot',8,1],[False,'rush_sonehot',8,1],[True,'rush',4,15]]

# [,"Driver_spatial_policy"]

import matplotlib.pyplot as plt
# x = embed
multiplier = 1
alpha=0.5
fig = plt.figure(tight_layout = True,figsize=(11,7))
days = [4,7,14]
cfg = ['Linear Setting','Nonlinear Setting I','Nonlinear Setting II']
for i in range(2):
    result = rdata[i]
    error = stdata[i]
    for j,cc in enumerate(cfg):

        x = top
        
        gs = gridspec.GridSpec(2,3)

        ax = fig.add_subplot(gs[i, j])        

#         ax = fig.add_subplot(2,len(cfg),i*len(cfg)+j+1)
#                 # result = np.zeros((len(top),len(cfg),len(deepset),4,ldata.shape[1]))
#         gs = gridspec.GridSpec(2,3)
        plt.grid(True)

        y1 = result[:,j,0,0]
        y2 = result[:,j,0,1]
        y3 = result[:,j,0,2]
        stdy3 = error[:,j,0,2]

        x1 = result[:,j,1,0]
        x2 = result[:,j,1,1]
        x3 = result[:,j,1,2]
        stdx3 = error[:,j,1,2]

        print('start plot!')
        print(y1)
        print(y3)

        # ax.plot(x,real_value,'--',color='red',label='real_value')
#         ax.plot(x,y1,'-o',color='green',label='mDR')
        # ax.plot(x,y2,'-o',color='blue',label='mIS')
        ax.tick_params(left = False, bottom = False)
        ax.plot(x,y3,color='darkred', markerfacecolor='none', marker='o', markersize =5,label = "Mean DE")
#         ax.plot(x,x1,'-o',color='navy',label='dDR')
        # ax.plot(x,x2,'-o',color='darkred',label='dIS')
        ax.plot(x,x3, color='darkgreen', markerfacecolor='none', marker='o', markersize =5,label = "Deep DE")

#         ax.fill_between(x,y3- multiplier* stdy3/np.sqrt(50),y3+ multiplier* stdy3/np.sqrt(50),color='lightcyan',alpha=alpha)    
#         ax.fill_between(x,x3- multiplier* stdx3/np.sqrt(50),x3+ multiplier* stdx3/np.sqrt(50),color='linen',alpha=alpha) 
        ax.fill_between(x,y3- multiplier* stdy3/np.sqrt(50),y3+ multiplier* stdy3/np.sqrt(50),color='darkred',alpha=alpha)    
        ax.fill_between(x,x3- multiplier* stdx3/np.sqrt(50),x3+ multiplier* stdx3/np.sqrt(50),color='darkgreen',alpha=alpha)  
        
# LU.plot(TDR5, color='magenta', markerfacecolor='none', marker='o', markersize =5,label = "Deep DR")
# LU.plot(TPLG5, color='limegreen', markerfacecolor='none', marker='o', markersize =5,label = "Deep PLG")
# LU.plot(FDR5, color='darkred', markerfacecolor='none', marker='o', markersize =5,label = "Mean DR")
# LU.plot(FPLG5, color='darkgreen', markerfacecolor='none', marker='o', markersize =5,label = "Mean PLG")        
        ax.set_title(' {}'.format(cc))
        
        if i==0:
            ax.set_xlabel("l=5")
        else:
            ax.set_xlabel("l=10")
        ax.set_xticks(x)
#         ax.set_ylim(0,0.5)
#         ax.set_xlabel(title)
        ax.set_xticklabels(x)
        ax.set_ylabel("MSE")
#         ax.legend(loc=1)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc = 'lower center', ncol =2, bbox_to_anchor = (0.5,-0.05))


fig.savefig('nondynamic.png',facecolor = "w", bbox_inches= 'tight')