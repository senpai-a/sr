import os
import pickle
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pickle

ea ='results/b500/error-angle.p'
es ='results/b500/error-strength.p'
ec ='results/b500/error-coherence.p'
ecent='results/b500_errcent/error-cent.p'
'''
ef=np.zeros((3,7,1000,1000))
with open(ea,'rb') as f:
    ea=pickle.load(f)
    ef[0,:,:,:]=ea
with open(es,'rb') as f:
    es=pickle.load(f)
    ef[1,:,:,:]=es
with open(ec,'rb') as f:
    ec=pickle.load(f)
    ef[2,:,:,:]=ec
'''
with open(ecent,'rb') as f:
    ecent=pickle.load(f)
i = 0
fig,axs=plt.subplots(6,sharex=True, sharey=True)
for clsid in range(6):
    for fl in range(1):
        i+=1
        ax1=axs[clsid]
        #ax2=axs[co,pt*2+1]
        plot1=ax1.matshow(np.log10(ecent[clsid,:,:]+1.))
        #plot2=ax2.matshow(len2[:,:,co,pt],cmap='autumn')
        #fig.colorbar(plot1,ax1,shrink = 0.9)


#fig.colorbar(plot2,ax=[axs[:,0]],location='left',shrink = 0.5)


plt.show()