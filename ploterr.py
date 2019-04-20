import os
import pickle
import sys
import numpy as np
from matplotlib import pyplot as plt

err1 = 'results/filter_91_err/classError.p'
len1 = 'results/filter_91_err/classCount.p'
err2 = 'results/filterEx2_err/classError.p'
len2 = 'results/filterEx2_err/classCount.p'
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
dmse = mse2 - mse1
print(dmse)
i = 0
fig,axs=plt.subplots(3,4,sharex=True, sharey=True)
for co in range(3):
    for pt in range(4):
        i+=1
        ax=axs[co,pt]
        plot1=ax.matshow(dmse[:,:,co,pt])

fig.colorbar(plot1,ax=[axs[:,:]],location='right',shrink = 0.5)
plt.tight_layout()
plt.show()