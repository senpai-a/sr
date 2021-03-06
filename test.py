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
from scipy.misc import imresize
from skimage import transform
from bicubic import bicubic2x,bicubic0_5x
import argparse
from testing import predict

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
    print("ignoring error:",e)

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

if args.groundTruth:
    classError = np.zeros((Qangle,Qstrength,Qcoherence,R*R))
    classCount = np.zeros((Qangle,Qstrength,Qcoherence,R*R))
    '''
    patchErrorDistAngle = np.zeros((7,1000,1000))
    patchErrorDistStrength = np.zeros((7,1000,1000))
    patchErrorDistCoherence = np.zeros((7,1000,1000))
    
    patchErrorDistCent = np.zeros((7,256,1000))
    classid=[(12,0,0),(11,2,1),(12,2,2),(17,2,0),(17,2,1),(6,2,1)]
    '''

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
        ycrcvorigin=imresize(ycrcv,1/2,interp='bicubic')
        cv2.imwrite('results/'+ args.output + '/' + os.path.splitext(os.path.basename(image))[0] + 'LR.png',
            cv2.cvtColor(ycrcvorigin, cv2.COLOR_YCrCb2BGR))        
    else:
        ycrcvorigin=cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)
    grayorigin = ycrcvorigin[:,:,0]

    '''
    # Normalized to [0,1]
    grayorigin = cv2.normalize(grayorigin.astype('float'), None, grayorigin.min()/255, grayorigin.max()/255, cv2.NORM_MINMAX)
    '''
    grayorigin=grayorigin.astype(float)/255.
    # Upscale (bilinear interpolation)
    heightLR, widthLR = grayorigin.shape
    upscaledLR = imresize(grayorigin,200,interp='bicubic').astype(float)/255.
    # Calculate predictHR pixels
    
    heightHR, widthHR = upscaledLR.shape
    predictHR = np.zeros((heightHR-2*margin, widthHR-2*margin))
    predict(upscaledLR,margin,predictHR,h,args.groundTruth,exQ,ycrcv,classError,classCount,weighting)    

    # Scale back to [0,255]
    predictHR = np.uint8(np.clip(predictHR.astype('float')*255., 0., 255.))
    '''
    cv2.imshow('LR',grayorigin)
    cv2.imshow('upscaledLR',upscaledLR)
    cv2.imshow('SR',predictHR)
    print('LR',grayorigin)
    print('upscaledLR',upscaledLR)
    print('SR',predictHR)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # Bilinear interpolation on CbCr field
    result = np.zeros((heightHR, widthHR, 3))
    result = imresize(ycrcvorigin,200,interp='bicubic')

    cv2.imwrite('results/'+ args.output + '/' + os.path.splitext(os.path.basename(image))[0] + 'Interp.png',
            cv2.cvtColor(np.uint8(result), cv2.COLOR_YCrCb2BGR))
    result[margin:heightHR-margin,margin:widthHR-margin,0] = predictHR

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

with open('results/'+ args.output + '/classCount.p','wb') as f:
    pickle.dump(classCount,f)
with open('results/'+ args.output + '/classError.p','wb') as f:
    pickle.dump(classError,f)
'''
with open('results/'+ args.output + '/error-angle.p','wb') as f:
    pickle.dump(patchErrorDistAngle,f)
with open('results/'+ args.output + '/error-strength.p','wb') as f:
    pickle.dump(patchErrorDistStrength,f)
with open('results/'+ args.output + '/error-coherence.p','wb') as f:
    pickle.dump(patchErrorDistCoherence,f)
with open('results/'+ args.output + '/error-cent.p','wb') as f:
    pickle.dump(patchErrorDistCent,f)'''
print('\r', end='')
print(' ' * 60, end='')
print('\rFinished.')
