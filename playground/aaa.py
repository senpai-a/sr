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
gt=cv2.cvtColor(cv2.imread(gt),cv2.COLOR_BGR2YCrCb)[:,:,0]
gth=hpf(gt)
gtl=gt-gth
h,w=gt.shape
n=w*h

srlist = []

for parent, dirnames, filenames in os.walk("SR_testing_datasets\\Set14_SR\\Set14\\image_SRF_2"):
    for filename in filenames:
        if filename.lower().startswith(('img_014_SRF_2_')):
            srlist.append(os.path.join(parent, filename))
print(srlist)
for sr_ in srlist:
    sr=cv2.cvtColor(cv2.imread(sr_),cv2.COLOR_BGR2YCrCb)[:,:,0]   
    srh=hpf(sr)
    srl=srh-srl
    errh=gth-srh
    errl=gtl-srl
    mseh=np.sum(np.power(errh,2))/n
    msel=np.sum(np.power(errl,2))/n
    psnrh=20*np.log10(255)-10*np.log10(mseh)
    psnrl=20*np.log10(255)-10*np.log10(msel)
    cv2.imwrite('hf_'+sr_,srh)
    cv2.imwrite('lf_'+sr_,srl)
    print(sr_,psnrh,psnrl)



    
