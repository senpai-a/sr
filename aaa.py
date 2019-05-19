import cv2
import numpy as np
import os
import pickle
import sys
import argparse
from math import floor, pi
from matplotlib import pyplot as plt
from scipy import interpolate
from skimage import transform
from hpf import hpf

gt="SR_testing_datasets\\Set14_SR\\Set14\\image_SRF_2\\img_014_SRF_2_HR.png"
gt=(cv2.cvtColor(cv2.imread(gt),cv2.COLOR_BGR2YCrCb)[:,:,0])
gt=gt.astype(float)
gth=hpf(gt)
gth=gth.astype(float)
gtl=gt-gth
h,w=gt.shape
n=w*h

srlist = []

for parent, dirnames, filenames in os.walk("SR_testing_datasets\\Set14_SR\\Set14SR"):
    for filename in filenames:
        srlist.append(os.path.join(parent, filename))
print(srlist)
for sr_ in srlist:
    sr=cv2.cvtColor(cv2.imread(sr_),cv2.COLOR_BGR2YCrCb)[:,:,0]
    sr=sr.astype(float)
    srh=hpf(sr).astype(float)
    srl=sr-srh
    errh=gth-srh
    errl=gtl-srl
    mseh=np.sum(np.power(errh,2))/n
    msel=np.sum(np.power(errl,2))/n
    psnrh=20*np.log10(255)-10*np.log10(mseh)
    psnrl=20*np.log10(255)-10*np.log10(msel)
    
    cv2.imwrite(sr_+'h.png',np.uint8(np.clip(srh,0.,255.)))
    cv2.imwrite(sr_+'l.png',np.uint8(np.clip(srl,0.,255.)))
    print(sr_,psnrh,psnrl)

cv2.imwrite("SR_testing_datasets\\Set14_SR\\Set14SR\\"+'gth.png',np.uint8(np.clip(gth,0.,255.)))
cv2.imwrite("SR_testing_datasets\\Set14_SR\\Set14SR\\"+'gtl.png',np.uint8(np.clip(gtl,0.,255.)))

    
