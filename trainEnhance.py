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

parser = argparse.ArgumentParser()
parser.add_argument("-sr", "--sr", help="Specify SR images folder")
parser.add_argument("-gt", "--gt", help="Specify GT images folder")
parser.add_argument("-s", "--selector", help="Specify selector file(PCA+8SVM)")
parser.add_argument("-o", "--output", help="File to save filter")
parser.add_argument("-ff", "--frefactor", help="Factor for fre SVC, spa SVC uses (1-ff).")

args = parser.parse_args()

patchSize=11
filterSize=patchSize*patchSize
margin = patchSize//2
W = np.diag(gaussian2d((9,9),2).ravel())

if args.ff:
    frefactor=float(args.ff)
else:
    frefactor=.3
spafactor=1-frefactor

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

with open(args.s,'rb') as f:
    (pcaL,svcspa,svcfre)=pickle.load(f)

#argmin||Qh-V||
Q=np.zeros((24,3,3,filterSize,filterSize))
V=np.zeros((24,3,3,filterSize))
h=np.zeros((24,3,3,filterSize))

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
    (h,w)=gt.shape
    if gt.shape!=sr.shape:
        if h%2==1:
            h-=1
        if w%2==1:
            w-=1
        gt=gt[0:h,0:w]
    
    if gt.shape!=sr.shape:
        print('dimensions do not match on image:',gtlist[i],srlist[i])
        print(gt.shape,sr.shape)
        continue

    srh=hpf(sr)
    gth=hpf(gt)
    patchi=0
    patchN=(h-2*margin) * (w-2*margin)
    for row in range(margin,w-margin):
        for col in range(margin,h-margin):
            
            print('\r',imagei,'/',imageN,' images|',
                '█'*(patchi*50//patchN),
                ' '*(50-patchi*50//patchN),'|',
                end='',sep='')
            sys.stdout.flush()
            
            if srh[row,col]<=20:
                continue

            srhpatch=srh[row-margin:row+margin+1,col-margin:col+margin+1]
            gthpixel=gth[row,col]

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
            
            A=np.matrix(srhpatch.ravel())
            ATA=A.T.dot(A)
            ATb=np.array(A.T.dot(gthpixel)).ravel()
            Q[angle,strength,coherence]+=ATA
            V[angle,strength,coherence]+=ATb

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

print('resolving filters...')
processi=0
processN=24*3*3
sys.stdout.flush()
for angle in range(24):
    for strength in range(3):
        for coherence in range(3):
            processi+=1
            print('\r|','█'*(patchi*50//patchN),
                ' '*(50-patchi*50//patchN),'|',
                end='',sep='')
            sys.stdout.flush()
            h[angle,strength,coherence] = cgls(Q[angle,strength,coherence],
                      V[angle,strength,coherence])

of='enhanceFilter.bin'
if args.output:
    of = args.output
with open(of,'wb') as f:
    pickle.dump(h,f)
