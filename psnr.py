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
for parent, dirnames, filenames in os.walk(args.gt):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            gtlist.append(os.path.join(parent, filename))
for parent, dirnames, filenames in os.walk(args.sr):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            srlist.append(os.path.join(parent, filename))

print(np.matrix([gtlist,srlist]).T)
if len(gtlist)!=len(srlist):
    print('error:file number does not match')
    exit()

pixelcount=0
error=0.
for i in range(len(gtlist)):
    gt=cv2.cvtColor(cv2.imread(gtlist[i]),cv2.COLOR_BGR2YCrCb)[:,:,0]
    sr=cv2.cvtColor(cv2.imread(srlist[i]),cv2.COLOR_BGR2YCrCb)[:,:,0]
    gth,gtw=gt.shape
    if gth%2==1:
        gth-=1
    if gtw%2==1:
        gtw-=1
    gt=gt[0:gth,0:gtw]
    if gt.shape!=sr.shape:
        print('dimensions do not match on image:',gtlist[i],srlist[i])
        print(gt.shape,sr.shape)
        continue
    gt=gt.astype('float')#/255.
    sr=sr.astype('float')#/255.
    err=np.power(gt-sr,2)
    error+=np.sum(err)
    pixelcount+=gtw*gth
mse=error/pixelcount
psnr=20*np.log10(255)-10*np.log10(mse)
print("PSNR=",psnr)
