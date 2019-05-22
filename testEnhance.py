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
from testEnhanceFunc import enhance


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Specify input SR images folder")
parser.add_argument("-s", "--selector", help="Specify selector file(PCA+8SVM)")
parser.add_argument("-o", "--output", help="Specify output folder")
parser.add_argument("-f", "--filter", help="Filter file.)")
args = parser.parse_args()

patchSize=11
filterSize=patchSize*patchSize
margin = patchSize//2
W = np.diag(gaussian2d((9,9),2).ravel())

with open(args.filter,'rb') as f:
    h=pickle.load(f)
with open(args.selector,'rb') as f:
    (pcaL,svc)=pickle.load(f)

try:
    os.mkdir('results/'+ args.output)
except Exception as e:
    print("ignoring error:",e)

imagelist=[]
for parent, dirnames, filenames in os.walk(args.input):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            imagelist.append(os.path.join(parent, filename))

imagei=0
imageN=len(imagelist)
for image in imagelist:
    imagei+=1
    origin=cv2.cvtColor(cv2.imread(image),cv2.COLOR_BGR2YCrCb)
    sr=origin[:,:,0]
    srh=hpf(sr).astype(float)
    srl=sr.astype(float)-srh
    ehh=srh
    mask=np.zeros(sr.shape)
    width,height=sr.shape    
    enhance(srh,ehh,mask,width,height,margin,pcaL,svc,h,imagei,imageN,patchSize,W)

    result=origin
    result[:,:,0]=np.uint8(np.clip(srl+ehh,0.,255.))    
    result=cv2.cvtColor(result,cv2.COLOR_YCrCb2BGR)
    cv2.imwrite('results/'+ args.output + '/' +\
        os.path.splitext(os.path.basename(image))[0] + '.png',result)
    cv2.imwrite('results/'+ args.output + '/' +\
        os.path.splitext(os.path.basename(image))[0] + 'mask.png',mask)