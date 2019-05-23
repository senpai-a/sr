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
from cgls import cgls
from ls import ls
from trainEnhanceFunc import stackQV

parser = argparse.ArgumentParser()
parser.add_argument("-sr", "--sr", help="Specify SR images folder")
parser.add_argument("-gt", "--gt", help="Specify GT images folder")
parser.add_argument("-s", "--selector", help="Specify selector file(PCA+4SVM)")
parser.add_argument("-o", "--output", help="File to save filter")
parser.add_argument('-qv','--QV',help="Specify QV file")
parser.add_argument('-c','--count',help="Specify count file")
#parser.add_argument("-ff", "--frefactor", help="Factor for fre SVC, spa SVC uses (1-ff).")

args = parser.parse_args()

patchSize=11
filterSize=patchSize*patchSize
margin = patchSize//2
W = np.diag(gaussian2d((9,9),2).ravel())

'''if args.ff:
    frefactor=float(args.ff)
else:    
    frefactor=.3
spafactor=1-frefactor'''
Q=np.zeros((24,3,3,filterSize,filterSize))
V=np.zeros((24,3,3,filterSize))
h=np.zeros((24,3,3,filterSize))
count=np.zeros((24,3,3,filterSize))
        
if args.QV:
    with open(args.QV,'rb') as f:
        (Q,V)=pickle.load(f)
    if args.count:
        with open(args.count,'rb') as f:
            count=pickle.load(f)
if args.gt and args.sr:
    with open(args.selector,'rb') as f:
        (pcaL,svc)=pickle.load(f)
    srlist=[]
    sl=[]
    gtlist=[]
    gl=[]
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
    imagei=0
    imageN=len(gtlist)
    print('collecting patches...')
    sys.stdout.flush()

    for i in range(len(gtlist)):
        imagei+=1
        j=-1
        try:
            j = sl.index(gl[i])
        except Exception as e:
            print('cannot find file with the same name in sr folder: '+gtlist[i])
            print(e)
            exit()
        if j==-1:
            print('cannot find file with the same name in sr folder: '+gtlist[i])
            exit()
        
        gt=cv2.cvtColor(cv2.imread(gtlist[i]),cv2.COLOR_BGR2YCrCb)[:,:,0]
        sr=cv2.cvtColor(cv2.imread(srlist[j]),cv2.COLOR_BGR2YCrCb)[:,:,0]
        (height,width)=gt.shape
        if gt.shape!=sr.shape:
            if height%2==1:
                height-=1
            if width%2==1:
                width-=1
            gt=gt[0:height,0:width]
        
        if gt.shape!=sr.shape:
            print('dimensions do not match on image:',gtlist[i],srlist[i])
            print(gt.shape,sr.shape)
            continue

        srh=hpf(sr).astype(float)/255.
        gth=hpf(gt).astype(float)/255.
        '''
        cv2.imshow('srh',srh)
        cv2.imshow('gth',gth) 
        print('srh',srh)   
        print('gth',gth)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        stackQV(Q,V,count,srh,gth,margin,width,height,imagei,imageN,patchSize,W,pcaL,svc)
    print('count',count)
    print('fliping samples...')
    sys.stdout.flush()

    Qextended = np.zeros((24, 3, 3, filterSize, filterSize))
    Vextended = np.zeros((24, 3, 3, filterSize))
    trans = np.zeros((filterSize, filterSize, 7))
    rotate = np.zeros((filterSize, filterSize))
    flip = np.zeros((filterSize,filterSize))
    for i in range(0, filterSize):
        i1 = i % patchSize
        i2 = int(i / patchSize)
        j = filterSize - patchSize + i2 - patchSize * i1
        rotate[j,i] = 1
        k = patchSize * (i2 + 1) - i1 - 1
        flip[k,i] = 1
    #get transfrom matrices
    for i in range(1, 8):
        i1 = i % 4
        i2 = int(i / 4)
        trans[:,:,i-1] = np.linalg.matrix_power(flip,i2).dot(np.linalg.matrix_power(rotate,i1))
    for angle in range(0, 24):
        for strength in range(0, 3):
            for coherence in range(0, 3):
                for m in range(1, 8):
                    m1 = m % 4
                    m2 = int(m / 4)
                    newangleslot = angle
                    if m2 == 1:
                        newangleslot = 24-angle-1
                    newangleslot = int(newangleslot-24/2*m1)
                    while newangleslot < 0:
                        newangleslot += 24
                    newQ = trans[:,:,m-1].T.dot(Q[angle,strength,coherence]).dot(trans[:,:,m-1])
                    newV = trans[:,:,m-1].T.dot(V[angle,strength,coherence])
                    Qextended[newangleslot,strength,coherence] += newQ
                    Vextended[newangleslot,strength,coherence] += newV
    Q += Qextended
    V += Vextended

of='enhanceFilter.bin'
if args.output:
    of = args.output
with open(of+"_QV",'wb') as f:
    pickle.dump((Q,V),f)
with open(of+'_count','wb') as f:
    pickle.dump(count,f)
print('resolving filters...')
processi=0
processN=24*3*3
sys.stdout.flush()
illsample=0
for angle in range(24):
    for strength in range(3):
        for coherence in range(3):
            processi+=1
            print('\r',processi,'/',processN,end='')
            sys.stdout.flush()
            if args.count or (args.sr and args.gt):
                if count[angle,strength,coherence]<10000:
                    h[angle,strength,coherence,60]=1
                    illsample+=1
                else:
                    h[angle,strength,coherence] = ls(Q[angle,strength,coherence],\
                        V[angle,strength,coherence],1)
            else:
                if np.sum(V[angle,strength,coherence])<1000:
                    h[angle,strength,coherence,60]=1
                    illsample+=1
                else:
                    h[angle,strength,coherence] = ls(Q[angle,strength,coherence],\
                        V[angle,strength,coherence],1)
print("\nclasses lack of sample:",illsample)
with open(of,'wb') as f:
    pickle.dump(h,f)
