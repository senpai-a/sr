import cv2
import numpy as np
import os
import pickle
import sys
from gaussian2d import gaussian2d
#from gettestargs import gettestargs
from hashkey import hashkey
from math import floor, pi
from matplotlib import pyplot as plt
from scipy import interpolate
from skimage import transform
from bicubic import bicubic2x,bicubic0_5x
import argparse
from testing import predict
from cgls import cgls
from ls import ls,ls2
from filterplot import filterplot


with open('q_b200','rb') as f:
    Q2=pickle.load(f)
with open('v_b200','rb') as f:
    V2=pickle.load(f)
with open('q_t91','rb') as f:
    Q1=pickle.load(f)
with open('v_t91','rb') as f:
    V1=pickle.load(f)
qq=Q1[7,2,1,1]
vv=V1[7,2,1,1]
qq2=Q2[7,2,1,1]
vv2=V2[7,2,1,1]

print('T91:Q',qq)
print('T91:V',vv)
print('B200:Q',qq2)
print('B200:V',vv2)
h1=np.reshape(ls(qq,vv,1), (11, 11))
h2=np.reshape(ls(qq2,vv2,1), (11, 11))
print('h1',h1)
print('h2',h2)
fig = plt.figure()
ax=fig.add_subplot(2,1,1)
ax.imshow(h1, interpolation='none', extent=[0,10,0,10])
ax.axis('off')
ax=fig.add_subplot(2,1,2)
ax.imshow(h2, interpolation='none', extent=[0,10,0,10])
ax.axis('off')
plt.show()


'''
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filter", help="Use file as filter")
parser.add_argument("-p", "--plot", help="Visualizing the process of RAISR image upscaling",
 action="store_true")
parser.add_argument("-e", "--extended", help="Use Extended Linear Mapping", action="store_true")
parser.add_argument("-o", "--output", help="output folder name")
parser.add_argument("-i", "--input", help="input folder name")
parser.add_argument("-gt", "--groundTruth", help="Use test images as ground truth (down scale them first)",
action="store_true")
parser.add_argument("-li", "--linear", help="Use bilinear for init",
action="store_true")
parser.add_argument("-ex2", "--ex2", help="Use normalized features for ExLM",
action="store_true")
parser.add_argument("-cv2", "--cv2", help="Use cv2 interpolation",
action="store_true")
args = parser.parse_args()

exQ=args.extended

# Define parameters
R = 2
patchsize = 11
gradientsize = 9
Qangle = 24
Qstrength = 3
Qcoherence = 3

if exQ:
    filterSize=patchsize*patchsize+4
else:
    filterSize=patchsize*patchsize

try:
    os.mkdir('results/'+ args.output)
except Exception as e:
    pass#print("\nignoring error:",e)

# Calculate the margin
maxblocksize = max(patchsize, gradientsize)
margin = floor(maxblocksize/2)
patchmargin = floor(patchsize/2)
gradientmargin = floor(gradientsize/2)

# Read filter from file
filtername = 'filter.p'
if args.filter:
    filtername = args.filter
with open(filtername, "rb") as fp:
    h = pickle.load(fp)

if h.shape[-1]!=filterSize:
    print('Error: filter size does not match.')
    print('extended=',exQ,'filter size is',h.shape[-1])
    exit()

# Matrix preprocessing
# Preprocessing normalized Gaussian matrix W for hashkey calculation
weighting = gaussian2d([gradientsize, gradientsize], 2)
weighting = np.diag(weighting.ravel())

# Get image list
tpath = 'test'
if args.input:
    tpath = args.input

imagelist = []
for parent, dirnames, filenames in os.walk(tpath):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            imagelist.append(os.path.join(parent, filename))

imagecount = 1
for image in imagelist:
    print('\r', end='')
    print(' ' * 60, end='')
    print('\rUpscaling image ' + str(imagecount) + ' of ' + str(len(imagelist)) + ' (' + image + ')')
    origin = cv2.imread(image)
    # Extract only the luminance in YCbCr
    ycrcv = cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)
    
    # Downscale
    if args.groundTruth:
        height, width = ycrcv[:,:,0].shape
        if height%2==1:
            height-=1
        if width%2==1:
            width-=1
        ycrcv = ycrcv[0:height,0:width,:]   
        ycrcvorigin=np.zeros((int(height/2),int(width/2),3))
        if args.cv2:
            ycrcvorigin=cv2.resize(ycrcv,(int(width/2),int(height/2)),interpolation=cv2.INTER_CUBIC)
        else:
            ycrcvorigin=np.uint8(np.clip(bicubic0_5x(ycrcv),0.,255.))
        cv2.imwrite('results/'+ args.output + '/' + os.path.splitext(os.path.basename(image))[0] + 'LR.png',
            cv2.cvtColor(ycrcvorigin, cv2.COLOR_YCrCb2BGR))
        
    else:
        ycrcvorigin=cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)
    grayorigin = ycrcvorigin[:,:,0]

    # Normalized to [0,1]
    grayorigin = cv2.normalize(grayorigin.astype('float'), None, grayorigin.min()/255, grayorigin.max()/255, cv2.NORM_MINMAX)
    
    # Upscale (bilinear interpolation)
    heightLR, widthLR = grayorigin.shape
    if args.linear:
        upscaledLR = cv2.resize(grayorigin,(widthLR*2,heightLR*2),interpolation=cv2.INTER_LINEAR)
    else:
        if args.cv2:
            upscaledLR = cv2.resize(grayorigin,(widthLR*2,heightLR*2),interpolation=cv2.INTER_CUBIC)
        else:
            upscaledLR = bicubic2x(grayorigin)
    heightHR, widthHR = upscaledLR.shape
    # Bilinear interpolation on CbCr field
    result = np.zeros((heightHR, widthHR, 3))
    result = np.clip(bicubic2x(ycrcvorigin),0.,255.)

    cv2.imwrite('results/'+ args.output + '/' + os.path.splitext(os.path.basename(image))[0] + 'Interp.png',
            cv2.cvtColor(np.uint8(result), cv2.COLOR_YCrCb2BGR))
    # Calculate predictHR pixels        
    predictHR = np.zeros((heightHR-2*margin, widthHR-2*margin))
    operationcount = 0
    totaloperations = (heightHR-2*margin) * (widthHR-2*margin)
    for row in range(margin, heightHR-margin):
        for col in range(margin, widthHR-margin):
            if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
                print('\r|', end='')
                print('#' * round((operationcount+1)*100/totaloperations/2), end='')
                print(' ' * (50 - round((operationcount+1)*100/totaloperations/2)), end='')
                print('|  ' + str(round((operationcount+1)*100/totaloperations)) + '%', end='')
                sys.stdout.flush()
            operationcount += 1
            # Get patch
            patch = upscaledLR[row-margin:row+margin+1, col-margin:col+margin+1]
            # Get gradient block
            gradientblock = patch #upscaledLR[row-gradientmargin:row+gradientmargin+1, col-gradientmargin:col+gradientmargin+1]
            # Calculate hashkey
            angle, strength, coherence, theta, lamda, u = hashkey(gradientblock, 24, weighting)
            patch = patch.ravel()
            if exQ:
                patch = np.concatenate((patch,np.array([theta,lamda,u,1.])),axis=None)
            # Get pixel type
            pixeltype = ((row-margin) % 2) * 2 + ((col-margin) % 2)
            predictHR[row-margin,col-margin] = patch.dot(h[angle,strength,coherence,pixeltype])
            pixelerror=result[row,col,0].astype('float')/255.-predictHR[row-margin,col-margin].astype('float')
            if pixelerror>0.8:
                print('-',(angle, strength, coherence, pixeltype))
            elif pixelerror <-0.8:
                print('+',(angle, strength, coherence, pixeltype))
            

    # Scale back to [0,255]
    predictHR = np.clip(predictHR.astype('float') * 255., 0., 255.)

    result = cv2.cvtColor(np.uint8(result), cv2.COLOR_YCrCb2RGB)

    cv2.imwrite('results/'+ args.output + '/' + os.path.splitext(os.path.basename(image))[0] + '.png',
     cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    imagecount += 1
    # Visualizing the process of RAISR image upscaling
    if args.plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 4, 1)
        ax.imshow(grayorigin, cmap='gray', interpolation='none')
        ax = fig.add_subplot(1, 4, 2)
        ax.imshow(upscaledLR, cmap='gray', interpolation='none')
        ax = fig.add_subplot(1, 4, 3)
        ax.imshow(predictHR, cmap='gray', interpolation='none')
        ax = fig.add_subplot(1, 4, 4)
        ax.imshow(result, interpolation='none')
        plt.show()

print('\r', end='')
print(' ' * 60, end='')
print('\rFinished.')
    '''