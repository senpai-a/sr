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
from numba import jit,prange
@jit(parallel=True)
def enhance(srh,ehh,mask,width,height,margin,pcaL,svc,h,imagei,imageN,patchSize,W):
    patchi=0
    patchN=(height-2*margin) * (width-2*margin)
    for row_ in range(width-2*margin):
        row=row_+margin
        print('\renhancing',imagei,'/',imageN,' images|',
                '█'*(patchi*50//patchN),
                ' '*(50-patchi*50//patchN),'|',
                end='',sep='')
        sys.stdout.flush()
        patchi+=height-2*margin
        for col_ in prange(height-2*margin):
            col=col_+margin          
            if srh[row,col]<=20:
                continue
            
            srhpatch=srh[row-margin:row+margin+1,col-margin:col+margin+1]
            #extract spa feature
            angle, strength, coherence, θ,λ,u = hashkey(srhpatch,24,W)
            selectAngle=(angle//3)%4
            #gy,gx=np.gradient(srhpatch)
            #sigma=np.cov(np.matrix([gx[1:-1,1:-1].ravel(),gy[1:-1,1:-1].ravel()]))
            #spa=np.concatenate((np.array([θ, λ, u]),sigma.ravel()))
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
            good=svc[selectAngle].predict([ff])

            if good==0:
                continue
            #set result
            srhpatchL=srhpatch.ravel()
            esti=srhpatchL.dot(h[angle,strength,coherence])
            ehh[row,col]=esti
            mask[row,col]=255