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
from cgls import cgls
from ls import ls
from numba import jit,prange

@jit(parallel=True)
def stackQV(Q,V,count,srh,gth,margin,w,h,imagei,imageN,patchSize,W,pcaL,svc):
    patchi=0
    patchN=(h-2*margin) * (w-2*margin)
    for col in range(margin,w-margin):        
        print('\r',imagei,'/',imageN,' images|',
                '█'*(patchi*50//patchN),
                ' '*(50-patchi*50//patchN),'|',
                end='',sep='')
        sys.stdout.flush()
        patchi+=h-2*margin
        for row in range(margin,h-margin):
            if srh[row,col]<=20:
                continue
            srhpatch=srh[row-margin:row+margin+1,col-margin:col+margin+1]
            gthpixel=gth[row,col]
            #extract spa feature
            angle, strength, coherence, θ,λ,u = hashkey(srhpatch,24,W)
            selectAngle=(angle//3)%4
            #gy,gx=np.gradient(srhpatch)
            #sigma=np.cov(np.matrix([gx[1:-1,1:-1].ravel(),gy[1:-1,1:-1].ravel()]))
            #spa=np.concatenate((np.array([λ, u]),sigma.ravel()))
            #extract fre feature
            spec = np.zeros((5,5,patchSize,patchSize)).astype(complex)
            orders = [0.6,0.7,0.8,0.9,1.]
            for xi in range(5):
                for yi in range(5):
                    spec[xi,yi,:,:]=frft2d(srhpatch,orders[xi],orders[yi])
            fre = zscore(np.absolute(spec).ravel())
            fre = pcaL[selectAngle].transform([fre])
            ff=np.concatenate((np.array([λ, u]),fre),axis=None)
            #select
            '''good=frefactor*svcfre[selectAngle].decision_function(fre)+\
                spafactor*svcspa[selectAngle].decision_function([spa])'''
            good=svc[selectAngle].predict([ff])
            if good==0:
                continue

            A=np.matrix(srhpatch.ravel())
            ATA=A.T.dot(A)
            ATb=np.array(A.T.dot(gthpixel)).ravel()
            Q[angle,strength,coherence]+=ATA
            V[angle,strength,coherence]+=ATb
            count[angle,strength,coherence]+=1