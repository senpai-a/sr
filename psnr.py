import cv2
import numpy as np
import os
import pickle
import sys
from gaussian2d import gaussian2d
from gettestargs import gettestargs
from hashkey import hashkey
from math import floor, pi
from matplotlib import pyplot as plt
import argparse

def getpsnrargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gt", "--gt", help="ground truth folder")
    parser.add_argument("-sr", "--sr", help="SR image folder")
    args = parser.parse_args()
    return args

args = getpsnrargs()

gtlist = []
srlist = []
gl=[]
sl=[]
for parent, dirnames, filenames in os.walk(args.gt):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.pbm', '.pgm', '.ppm', '.tif')):
            gtlist.append(os.path.join(parent, filename))
            gl.append(filename[:-4])
for parent, dirnames, filenames in os.walk(args.sr):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.pbm', '.pgm', '.ppm', '.tif')):
            srlist.append(os.path.join(parent, filename))
            sl.append(filename[:-4])

#print(np.matrix([gtlist,srlist]).T)
if len(gtlist)!=len(srlist):
    print('error:file number does not match')
    print(gtlist,srlist)
    exit()

pixelcount=0
error=0.
for i in range(len(gtlist)):
    if gl[i]!=sl[i]:
        print('filename dose not match. Quiting.')
        exit()
    #gt=cv2.cvtColor(cv2.imread(gtlist[i]),cv2.COLOR_BGR2YCrCb)[:,:,0]
    #sr=cv2.cvtColor(cv2.imread(srlist[i]),cv2.COLOR_BGR2YCrCb)[:,:,0]
    gt=cv2.imread(gtlist[i])[:,:,0:3]
    sr=cv2.imread(srlist[i])[:,:,0:3]
    (gth,gtw,ch)=gt.shape
    if gt.shape!=sr.shape:
        if gth%2==1:
            gth-=1
        if gtw%2==1:
            gtw-=1
        gt=gt[0:gth,0:gtw,:]

    if gt.shape!=sr.shape:
        print('dimensions do not match on image:',gtlist[i],srlist[i])
        print(gt.shape,sr.shape)
        continue
    #gt=gt.astype('double')#/255.
    #sr=sr.astype('double')#/255.
    err=np.power(gt-sr,2)
    #print(gt-sr)
    #print(err)
    error+=np.sum(err)
    pixelcount+=gtw*gth*3
mse=error/pixelcount
psnr=20*np.log10(255)-10*np.log10(mse)
print("PSNR=",psnr)
