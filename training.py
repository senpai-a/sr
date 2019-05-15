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


# Calculate A'A, A'b and push them into Q, V
@jit
def collectQV(lr,hr,patchsize,Q,V,W):
    margin = floor(patchsize/2)
    height,width=lr.shape
    for row in range(margin, height-margin):
        for col in range(margin, width-margin):            
            # Get patch
            patch = lr[row-margin:row+margin+1, col-margin:col+margin+1]
            # Get gradient block
            gradientblock = patch
            # Calculate hashkey
            angle, strength, coherence,theta,lamda,u=hashkey(gradientblock, 24, W)
            patch = np.ravel(patch)#flatten by row
            patch = np.matrix(patch)#row 
            # Get pixel type
            pixeltype = ((row-margin) % 2) * 2 + ((col-margin) % 2)
            # Get corresponding HR pixel
            pixelHR = hr[row,col]
            # Compute A'A and A'b                
            ATA = np.dot(patch.T, patch)
            ATb = np.dot(patch.T, pixelHR)
            ATb = np.array(ATb).ravel()
            # Compute Q and V
            Q[angle,strength,coherence,pixeltype] += ATA
            V[angle,strength,coherence,pixeltype] += ATb
            
#@njit(parallel=True)
@jit
def collectQVrotflip(lr,hr,patchsize,Q,V,W,P):
    margin = floor(patchsize/2)
    height,width=lr.shape
    filterSize=patchsize*patchsize
    for row in range(margin, height-margin):
        for col in range(margin, width-margin):            
             # Get patch
            patch = lr[row-margin:row+margin+1, col-margin:col+margin+1]
            # Get gradient block
            gradientblock = patch
            # Calculate hashkey
            angle, strength, coherence,theta,lamda,u = hashkey(gradientblock, 24, W)
            patch = np.ravel(patch)#flatten by row
            patch = np.matrix(patch)#row 
            # Get pixel type
            pixeltype = ((row-margin) % 2) * 2 + ((col-margin) % 2)
            # Get corresponding HR pixel
            pixelHR = hr[row,col]
            for i in range(1, 8):
                rot = i % 4
                fli = floor(i / 4)
                newangleslot = angle
                newtheta = theta
                if fli == 1:
                    newangleslot = 24-angle-1
                    newtheta = pi - theta
                newangleslot = int(newangleslot-24/2*rot)
                while newangleslot < 0:
                    newangleslot += 24
                newtheta = theta - rot*pi/2
                while newtheta < 0:
                    newtheta += pi
                
                patch_=np.zeros(filterSize)
                patch_=patch.dot(P[:,:,i-1])
                f=np.array([newtheta,lamda,u,1.])                    
                patch_=np.concatenate((np.array(patch_),f),axis=None)
                patch_=np.matrix(patch_)
                ATA=np.dot(patch_.T,patch_)
                ATb=np.dot(patch_.T,pixelHR)
                ATb=np.array(ATb).ravel()
                Q[newangleslot,strength,coherence,pixeltype] += ATA
                V[newangleslot,strength,coherence,pixeltype] += ATb

#@njit(parallel=True)
@jit
def rotflipQV(Q,V,P):
    Qextended = np.zeros(Q.shape)
    Vextended = np.zeros(V.shape)
    for pixeltype in range(4):
        for angle in range(24):
            for strength in range(3):
                for coherence in range(3):
                    for m in range(1, 8):
                        m1 = m % 4
                        m2 = floor(m / 4)
                        newangleslot = angle
                        if m2 == 1:
                            newangleslot = 24-angle-1
                        newangleslot = int(newangleslot-24/2*m1)
                        while newangleslot < 0:
                            newangleslot += 24
                        newQ = P[:,:,m-1].T.dot(Q[angle,strength,coherence,pixeltype]).dot(P[:,:,m-1])
                        newV = P[:,:,m-1].T.dot(V[angle,strength,coherence,pixeltype])
                        Qextended[newangleslot,strength,coherence,pixeltype] += newQ
                        Vextended[newangleslot,strength,coherence,pixeltype] += newV
    Q += Qextended
    V += Vextended

@jit(parallel=False)
def resolvefilters(Q,V,h):
    fin=0
    for pixeltype in range(4):
        for angle in range(24):
            for strength in range(3):
                for coherence in range(3):
                    print('\r',' '*20,'\r',fin,'/864',set='',end='')
                    sys.stdout.flush()
                    h[angle,strength,coherence,pixeltype] = cgls(Q[angle,strength,coherence,pixeltype],
                        V[angle,strength,coherence,pixeltype])
                    fin+=1
