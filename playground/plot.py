import os
import pickle
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pickle

dat = 'coStConut_B500cu.p'
with open(dat,'rb') as f:
    dat=pickle.load(f)
'''
i = 0
fig,axs=plt.subplots(1,3,sharex=True, sharey=True)
for co in range(3):
    i+=1
    ax1=axs[co]
    #ax2=axs[co,pt*2+1]
    plot1=ax1.matshow(dat[:,:,co,0])
    #plot2=ax2.matshow(len2[:,:,co,pt],cmap='autumn')

fig.colorbar(plot1,ax=[axs[:]],location='right',shrink = 0.9)
#fig.colorbar(plot2,ax=[axs[:,0]],location='left',shrink = 0.5)
'''
'''
plot1=plt.matshow(np.log10(dat[:,:1000]+1))
plt.colorbar(plot1)
print(np.sum(dat),np.amax(dat),np.amin(dat))
'''

co=np.zeros(1000)
st=np.zeros(10000)
cocdf=np.zeros(1000)
stcdf=np.zeros(2000)
for i in range(0,1000):
    co[i] = np.sum(dat[i,:])
for i in range(0,10000):
    st[i]=np.sum(dat[:,i])
for i in range(0,1000):
    cocdf[i]=np.sum(co[:i+1])
for i in range(0,2000):
    stcdf[i]=np.sum(st[0:i+1])
plot1=plt.bar(range(2000),stcdf[:],width=1)
#plot2=plt.plot(st)
mi=np.amin(cocdf)
ma=np.amax(cocdf)
cocdf = (cocdf-mi)/(ma-mi)
mi=np.amin(stcdf)
ma=np.amax(stcdf)
cocdf = (stcdf-mi)/(ma-mi)
with open("cocdf.p",'wb') as f:
    pickle.dump(cocdf,f)
with open("stcdf.p",'wb') as f:
    pickle.dump(stcdf,f)

plt.show()