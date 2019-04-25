import os
import pickle
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

err1 = 'results/t91onB100/classError.p'
len1 = 'results/t91onB100/classCount.p'
err2 = 'results/t91onB100Ex/classError.p'
len2 = 'results/t91onB100Ex/classCount.p'
with open(err1,'rb') as f:
    err1 = pickle.load(f)
with open(len1,'rb') as f:
    len1 = pickle.load(f)
with open(err2,'rb') as f:
    err2 = pickle.load(f)
with open(len2,'rb') as f:
    len2 = pickle.load(f)

mse1 = err1.astype('float')/len1
mse2 = err2.astype('float')/len2
dmse = (mse2 - mse1).astype('float')
pos = np.ma.masked_less(dmse,0)
neg = np.ma.masked_greater(dmse,0)
'''
ma=dmse.max()
mi=dmse.min()
rng=ma-mi
plvl=np.round(ma*1024./rng)
nlvl=1024-plvl
pos=cm.get_cmap('viridis',1024)
neg=cm.get_cmap('Reds',1024)


colorlist=np.vstack((neg(np.linspace(0.5, 1, nlvl)),
    pos(np.linspace(0, 1, plvl))))
colormap=ListedColormap(colorlist)
'''

#print(dmse[:,:,0,0])
i = 0
fig,axs=plt.subplots(3,4,sharex=True, sharey=True)
for co in range(3):
    for pt in range(4):
        i+=1
        ax1=axs[co,pt]
        #ax2=axs[co,pt*2+1]
        plot1=ax1.matshow(len1[:,:,co,pt])
        #plot2=ax2.matshow(len2[:,:,co,pt],cmap='autumn')
print(np.sum(len1),np.sum(len2))
fig.colorbar(plot1,ax=[axs[:,3]],location='right',shrink = 0.5)
#fig.colorbar(plot2,ax=[axs[:,0]],location='left',shrink = 0.5)

plt.show()