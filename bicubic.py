import argparse
import cv2
import numpy as np
import os
import pickle
import sys
from gaussian2d import gaussian2d
from hashkey import hashkey
from math import floor, pi
from matplotlib import pyplot as plt
from scipy import interpolate
from skimage import transform
from numba import njit, prange

def bicubic2x(mat):
    mch=False
    if len(mat.shape)==3:
        h,w,ch=mat.shape
        mch=True
    if mch:
        ret = np.zeros((h*2,w*2,ch))
        for i in range(ch):
            ret[:,:,i]=_bicubic2x(mat[:,:,i])
        return ret
    else:
        return _bicubic2x(mat)

#@njit(parallel=True)
def _bicubic2x(mat):
    h,w=mat.shape
    heightgridLR = np.linspace(0,h-1,h)
    widthgridLR = np.linspace(0,w-1,w)
    heightgridHR = np.linspace(0,h-0.5,h*2)
    widthgridHR = np.linspace(0,w-0.5,w*2)
    interp=interpolate.interp2d(widthgridLR, heightgridLR, mat, kind='cubic')
    return interp(widthgridHR,heightgridHR)

def bicubic0_5x(mat):
    mch=False
    if len(mat.shape)==3:
        h,w,ch=mat.shape
        mch=True
    if mch:        
        ret = np.zeros((int(h/2),int(w/2),ch))
        for i in range(ch):
            ret[:,:,i]=_bicubic0_5x(mat[:,:,i])
        return ret
    else:
        return _bicubic0_5x(mat)

#@njit(parallel=True)
def _bicubic0_5x(mat):
    h,w=mat.shape
    heightgridHR = np.linspace(0,h-1,h)
    widthgridHR = np.linspace(0,w-1,w)
    heightgridLR = np.linspace(0,h-2,h/2)
    widthgridLR = np.linspace(0,w-2,w/2)
    interp=interpolate.interp2d(widthgridHR, heightgridHR, mat, kind='cubic')
    return interp(widthgridLR,heightgridLR)