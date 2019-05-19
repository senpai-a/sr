import cv2
import numpy as np
import os
import pickle
import sys
from cgls import cgls
from ls import ls
from filterplot import filterplot
from gaussian2d import gaussian2d
import argparse
from hashkey import hashkey
from math import floor, pi
from matplotlib import pyplot as plt
from scipy import interpolate
from skimage import transform
from bicubic import bicubic2x,bicubic0_5x
from numba import njit, prange, jit

@jit
def predict(upscaledLR,margin,predictHR,h,gt,exQ,ycrcv,classError,classCount,W):
    heightHR,widthHR=upscaledLR.shape
    operationcount = 0
    totaloperations = (heightHR-2*margin) * (widthHR-2*margin)
    for row in range(margin, heightHR-margin):
        if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
            print('\r|', end='')
            print('#' * round((operationcount+1)*100/totaloperations/2), end='')
            print(' ' * (50 - round((operationcount+1)*100/totaloperations/2)), end='')
            print('|  ' + str(round((operationcount+1)*100/totaloperations)) + '%', end='')
            sys.stdout.flush()
        for col in range(margin, widthHR-margin):
            operationcount += 1
            # Get patch
            patch = upscaledLR[row-margin:row+margin+1, col-margin:col+margin+1]
            # Get gradient block
            gradientblock = patch #upscaledLR[row-gradientmargin:row+gradientmargin+1, col-gradientmargin:col+gradientmargin+1]
            # Calculate hashkey
            angle, strength, coherence, theta, lamda, u = hashkey(gradientblock, 24, W)
            patch = patch.ravel()
            '''if args.ex2:
                cocdf=np.zeros(1000)
                stcdf=np.zeros(2000)
                with open("cocdf.p","rb") as f:
                    cocdf=pickle.load(f)
                with open("stcdf.p","rb") as f:
                    stcdf=pickle.load(f)
                theta = (theta - angle*pi/24)/(pi/24)
                lamda = stcdf[int(lamda*1000)]
                u = cocdf[int(u*1000)]'''
            if exQ:
                patch = np.concatenate((patch,np.array([theta,lamda,u,1.])),axis=None)
            # Get pixel type
            pixeltype = ((row-margin) % 2) * 2 + ((col-margin) % 2)
            predictHR[row-margin,col-margin] = patch.dot(h[angle,strength,coherence,pixeltype])
            if gt:
                pixelerror=ycrcv[row,col,0].astype('float')/255.-predictHR[row-margin,col-margin].astype('float')
                classCount[angle,strength,coherence,pixeltype]+=1
                classError[angle,strength,coherence,pixeltype]+=pixelerror*pixelerror
                '''
                tslot=int((theta - angle*pi/24)/(pi/24)*1000)
                lslot=int(lamda*10000)
                if lslot>=1000:
                    lslot=999
                try:
                    uslot=int(u*1000)
                except Exception as e:
                    continue
                if uslot>=1000:
                    uslot=999
                eslot=int((pixelerror+.5)*1000)
                if eslot<0:
                    eslot=0
                if eslot>999:
                    eslot=999
                    
                cent=int(upscaledLR[row,col]*255)
                if cent<0:
                   cent=0
                if cent>255:
                   cent=255
                try:
                    cid=classid.index((angle,strength,coherence))
                except Exception as e:
                    cid=-1
                if cid!=-1:
                    
                    patchErrorDistAngle[cid,tslot,eslot]+=1
                    patchErrorDistStrength[cid,lslot,eslot]+=1
                    patchErrorDistCoherence[cid,uslot,eslot]+=1
                    patchErrorDistCent[cid,cent,eslot]+=1'''
