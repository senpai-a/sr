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

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Specify input SR images folder")
parser.add_argument("-s", "--selector", help="Specify selector file(PCA+8SVM)")
parser.add_argument("-o", "--output", help="Specify output folder")
parser.add_argument("-f", "--filter", help="Filter file.(frefactor,h))")
args = parser.parse_args()

patchSize=11
filterSize=patchSize*patchSize
margin = patchSize//2
W = np.diag(gaussian2d((9,9),2).ravel())

with open(args.filter,'rb') as f:
    (frefactor,h)=pickle.load(f)
with open(args.s,'rb') as f:
    (pcaL,svcspa,svcfre)=pickle.load(f)

spafactor=1-frefactor

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
    sr=cv2.cvtColor(cv2.imread(image),cv2.COLOR_BGR2YCrCb)[:,:,0]
    srh=hpf(sr)
    srl=sr-srh
    ehh=srh
    w,h=sr.shape

    patchi=0
    patchN=(h-2*margin) * (w-2*margin)
    for row in range(margin,w-margin):
        for col in range(margin,h-margin):
            print('\renhancing',imagei,'/',imageN,' images|',
                '█'*(patchi*50//patchN),
                ' '*(50-patchi*50//patchN),'|',
                end='',sep='')
            sys.stdout.flush()
            if srh[row,col]<=20:
                continue
            
            srhpatch=srh[row-margin:row+margin+1,col-margin:col+margin+1]
            #extract spa feature
            angle, strength, coherence, θ,λ,u = hashkey(srhpatch,24,W)
            selectAngle=(angle//3)%4
            gy,gx=np.gradient(srhpatch)
            sigma=np.cov(np.matrix([gx[1:-1,1:-1].ravel(),gy[1:-1,1:-1].ravel()]))
            spa=np.concatenate((np.array([θ, λ, u]),sigma.ravel()))
            #extract fre feature
            spec = np.zeros((5,5,patchSize,patchSize)).astype(complex)
            orders = [0.6,0.7,0.8,0.9,1.]
            for xi in range(5):
                for yi in range(5):
                    spec[xi,yi,:,:]=frft2d(srhpatch,orders[xi],orders[yi])
            fre = zscore(np.absolute(spec).ravel())
            fre = pcaL[selectAngle].transform(fre)
            #select
            good=frefactor*svcfre[selectAngle].predict(fre)+\
                spafactor*svcspa[selectAngle].predict(spa)

            if good<.5:
                continue
            #set result
            ehh[row,col]=np.matrix(srhpatch.ravle()).dot(h[angle,strength,coherence])

    result=origin
    result[:,:,0]=np.uint8(np.clip((srl+ehh).astype(float),0.,255.))    
    result=cv2.cvtColor(result,cv2.COLOR_YCrCb2BGR)
    cv2.imwrite('results/'+ args.output + '/' +\
        os.path.splitext(os.path.basename(image))[0] + '.png',result)