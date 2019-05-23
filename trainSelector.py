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
parser.add_argument("-i", "--input", help="Specify training set")
parser.add_argument("-o", "--output", help="File to save selector")
parser.add_argument("-d","--data",help='Training data file')
args = parser.parse_args()

W = np.diag(gaussian2d((9,9),2).ravel())
patchSize=11
gradientSize=9
patchMargin=patchSize//2
gradientMargin=gradientSize//2
margin=max(patchMargin,gradientMargin)

markCount = 200
patchSelect=np.zeros((4,markCount,patchSize,patchSize))
fspa=np.zeros((4,markCount,2))#λ,u,σxx,σxy,σyx,σyy
ffre=np.zeros((4,markCount,20))
Ffre=np.zeros((4,markCount,patchSize*patchSize*25))
fff=np.zeros((4,markCount,22))
mk=np.zeros((4,markCount))
pcaL=[PCA(n_components=20),PCA(n_components=20),PCA(n_components=20),PCA(n_components=20)]

if args.input:#use image and maunally marking
    imagelist = []
    for parent, dirnames, filenames in os.walk(args.input):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(parent, filename))

    Vsize  = 10000
    patchV = np.zeros((Vsize,4,patchSize,patchSize))
    Vid = np.zeros(4).astype(int)

    print('Collecting patches...')
    imcount=0
    for img in imagelist:
        imcount+=1
        im=cv2.imread(img)
        im=cv2.cvtColor(im,cv2.COLOR_BGR2YCrCb)[:,:,0]
        imh=hpf(im)
        w,h=imh.shape
        #cv2.imshow('',np.uint8(imh))
        #cv2.waitKey(0)
        #print(im.shape)
        #print(imh.shape)
        processi=0
        processma=(h-2*margin)*(w-2*margin)
        #collect all patched with central HF energy>20
        for row in range(margin,w-margin):
            print('\r',imcount,'/',len(imagelist),' images',end='|')
            print('█'*(processi*50//processma),end='')
            print(' '*(50-processi*50//processma),end='|')
            sys.stdout.flush()
            for col in range(margin,h-margin):
                processi+=1            
                if imh[row,col]<=20:
                    continue

                #gradientblock = imh[row-gradientMargin:row+gradientMargin+1,col-gradientMargin:col+gradientMargin+1]
                patch = imh[row-margin:row+margin+1,col-margin:col+margin+1]
                angle, strength, coherence, theta, lamda, u = hashkey(patch,8,W)
                '''print(angle,theta*180/3.14)
                cv2.imshow('',cv2.resize(np.uint8(patch),(110,110),interpolation=cv2.INTER_NEAREST))
                cv2.waitKey(0)
                cv2.destroyAllWindows()  '''         
                #get angle 0..3
                angle=angle%4
                patchV[Vid[angle],angle,:,:]=patch

                Vid[angle]+=1
                if Vid[angle]==Vsize: #realloc
                    Vsize*=2
                    patchV.resize((Vsize,4,patchSize,patchSize))
    print('Fin')
    
    
    processi=0
    processma=4*markCount
    print("sample count:",Vid)
    if(np.amin(Vid)<markCount):
        print('No enough samples.Quiting')
        exit()
    #mark patches manually
    for angle in range(4):
        selectId=np.random.permutation(Vid[angle])[:markCount]
        patchSelect[angle,:,:,:]=patchV[selectId,angle,:,:]
        i=0
        while i < markCount:
            processi=angle*markCount+i
            print('\r',processi,'/',processma,'samples marked',end='')
            sys.stdout.flush()

            p=np.uint8(patchSelect[angle,i,:,:])
            #print(p)
            showp=cv2.resize(p,(0,0),fx=55,fy=55,interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Please mark sample by pressing 0 for bad or 1 for good.q:abort|space:rollback",showp)
            mark=-1
            rb=False
            while mark==-1:
                k=cv2.waitKey(0)
                if k==ord('0'):
                    mark=0
                elif k==ord('1'):
                    mark=1
                elif k==ord('q'):
                    print('\nAborting\n')
                    exit()
                elif k==ord('p'):
                    print(p)
                elif k==ord(' '):
                    i-=1
                    rb=True
                    break        
            cv2.destroyAllWindows()
            if rb:
                continue
            else:
                mk[angle,i]=mark        
                i+=1

    print('Saving trianing data...',end='')
    with open(args.output+'_traindata','wb') as f:
        pickle.dump((patchSelect,mk),f)
    print('saved.')

elif args.data:#use marked data
    with open(args.data,'rb') as f:
        (patchSelect,mk)=pickle.load(f)

print('extracting features...')
#extract features
processi=0
processma=4*markCount
for angle in range(4):
    for i in range(markCount):
        processi+=1
        print('\r',processi,'/',processma,'samples processed',end='')
        sys.stdout.flush()
        patch=patchSelect[angle,i,:,:]/255.
        aaaa, strength, coherence, θ, λ, u = hashkey(patch,24,W)
        gy,gx=np.gradient(patch)
        sigma=np.cov(np.matrix([gx[1:-1,1:-1].ravel(),gy[1:-1,1:-1].ravel()]))
        spa=np.concatenate((np.array([λ, u]),sigma.ravel()))
        fspa[angle,i,:]=np.array([λ, u])
        #fspa[angle,i,:]=spa

        spec = np.zeros((5,5,patchSize,patchSize)).astype(complex)
        orders = [0.6,0.7,0.8,0.9,1.]
        for xi in range(5):
            for yi in range(5):
                spec[xi,yi,:,:]=frft2d(patch,orders[xi],orders[yi])
        Ffre[angle,i,:] = zscore(np.absolute(spec).ravel())

    pcaL[angle].fit(Ffre[angle,:,:])
    ffre[angle,:,:]=pcaL[angle].transform(Ffre[angle,:,:])
fff=np.concatenate((fspa,ffre),axis=2)

print('training svc...')
svcspa=[svm.SVC(kernel='linear'),svm.SVC(kernel='linear'),svm.SVC(kernel='linear'),svm.SVC(kernel='linear')]
svcfre=[svm.SVC(kernel='linear'),svm.SVC(kernel='linear'),svm.SVC(kernel='linear'),svm.SVC(kernel='linear')]
svc=[svm.SVC(kernel='linear',C=100),
     svm.SVC(kernel='linear',C=100),
     svm.SVC(kernel='linear',C=100),
     svm.SVC(kernel='linear',C=100)]
for angle in range(4):
    '''
    print('\r',angle*2+1,'/12 trained.',end='',sep='')
    sys.stdout.flush()
    svcspa[angle].fit(fspa[angle,:,:],mk[angle,:])

    print('\r',angle*2+2,'/12 trained.',end='',sep='')
    sys.stdout.flush()
    svcfre[angle].fit(ffre[angle,:,:],mk[angle,:])

    print('\r',angle+1,'/4 trained.',end='',sep='')
    sys.stdout.flush()
    svc[angle].fit(fff[angle,:,:],mk[angle,:])'''
    print('\r',angle+1,'/4 trained.',end='',sep='')
    svc[angle].fit(fff[angle,:,:],mk[angle,:])

print('Dumping SVMs')
with open(args.output,'wb') as f:
    pickle.dump((pcaL,svc),f)
print('Fin')