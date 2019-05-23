import cv2
import numpy as np
import os
import pickle
import sys
from gaussian2d import gaussian2d
#from gettestargs import gettestargs
from hashkey import hashkey
from math import floor, pi
from matplotlib import pyplot as plt
from scipy import interpolate
from skimage import transform
from bicubic import bicubic2x,bicubic0_5x
import argparse
from testing import predict
from cgls import cgls
from ls import ls,ls2
from filterplot import filterplot


with open('q_b200final','rb') as f:
    Q=pickle.load(f)
with open('v_b200final','rb') as f:
    V=pickle.load(f)
with open('b200final','rb') as f:
    h=pickle.load(f)
qq=Q[12,2,2,0]
vv=V[12,2,2,0]
hh=h[12,2,2,0]

print('B200:Q',qq)
print('B200:V',vv)
h1=np.reshape(ls(qq,vv,-1), (11, 11))
h2=np.reshape(hh, (11, 11))
print('h1',h1)
print('hh',hh)
fig = plt.figure()
ax=fig.add_subplot(1,2,1)
ax.imshow(h1, interpolation='none', extent=[0,10,0,10])
ax.axis('off')
ax=fig.add_subplot(1,2,2)
ax.imshow(h2, interpolation='none', extent=[0,10,0,10])
ax.axis('off')
plt.show()

