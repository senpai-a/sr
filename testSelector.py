import cv2
import numpy as np
from hpf import hpf
from hashkey import hashkey
from gaussian2d import gaussian2d
from frft import frft2d
import argparse
import os
import sys
import pickle
from scipy.stats import zscore
from sklearn import svm
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--selector", help="selector file")
parser.add_argument("-d", "--data", help="testing data")
args = parser.parse_args()

with open(args.selector,'rb') as f:
    (pcaL,svc)=pickle.load(f)

#patchSelect=np.zeros((4,markCount,patchSize,patchSize))
#mk=np.zeros((4,markCount))
with open(args.data,'rb') as f:
    (patchSelect,mk)=pickle.load(f)
markCount=mk.shape[-1]
patchSize=patchSelect.shape[-1]
W = np.diag(gaussian2d((9,9),2).ravel())

faccu=0
accu=0
testn=0
for ff in range(1):
    frefactor=ff*0.1
    #print('frefactor=',frefactor)
    for angle in range(4):
        for i in range(markCount):
            patch=patchSelect[angle,i,:,:]
            aaa, strength, coherence, θ,λ,u = hashkey(patch,24,W)
            selectAngle=(aaa//3)%4
            gy,gx=np.gradient(patch)
            sigma=np.cov(np.matrix([gx[1:-1,1:-1].ravel(),gy[1:-1,1:-1].ravel()]))
            spa=np.concatenate((np.array([λ, u]),sigma.ravel()))
            #extract fre feature
            spec = np.zeros((5,5,patchSize,patchSize)).astype(complex)
            orders = [0.6,0.7,0.8,0.9,1.]
            for xi in range(5):
                for yi in range(5):
                    spec[xi,yi,:,:]=frft2d(patch,orders[xi],orders[yi])
            fre = zscore(np.absolute(spec).ravel())
            fre = pcaL[selectAngle].transform([fre])
            ff=np.concatenate((spa,fre),axis=None)
            #select            
            m=svc[selectAngle].predict([ff])[0]
            tm=mk[angle,i]
            testn+=1
            if m==tm:
                accu+=1
            print('\raccu=',accu/testn,end='')
            sys.stdout.flush()
    print('')