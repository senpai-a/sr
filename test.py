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
from scipy import interpolate
from skimage import transform

args = gettestargs()
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
        ycrcvorigin[:,:,0] = transform.resize(ycrcv[:,:,0], (int(height/2),int(width/2)),order=3, mode='reflect', anti_aliasing=False)
        ycrcvorigin[:,:,1] = transform.resize(ycrcv[:,:,1], (int(height/2),int(width/2)),order=3, mode='reflect', anti_aliasing=False)
        ycrcvorigin[:,:,2] = transform.resize(ycrcv[:,:,2], (int(height/2),int(width/2)),order=3, mode='reflect', anti_aliasing=False)
        ycrcvorigin=np.uint8(np.clip(ycrcvorigin.astype('float')*255.,0.,255.))
    else:
        ycrcvorigin=cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)
    grayorigin = ycrcvorigin[:,:,0]

    # Normalized to [0,1]
    grayorigin = cv2.normalize(grayorigin.astype('float'), None, grayorigin.min()/255, grayorigin.max()/255, cv2.NORM_MINMAX)
    
    # Upscale (bilinear interpolation)
    heightLR, widthLR = grayorigin.shape    
    heightgridLR = np.linspace(0,heightLR-1,heightLR)
    widthgridLR = np.linspace(0,widthLR-1,widthLR)
    bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, grayorigin, kind='linear')
    heightgridHR = np.linspace(0,heightLR-0.5,heightLR*2)
    widthgridHR = np.linspace(0,widthLR-0.5,widthLR*2)
    upscaledLR = bilinearinterp(widthgridHR, heightgridHR)
    # Calculate predictHR pixels
    
    heightHR, widthHR = upscaledLR.shape
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
            patch = upscaledLR[row-patchmargin:row+patchmargin+1, col-patchmargin:col+patchmargin+1]
            patch = patch.ravel()
            # Get gradient block
            gradientblock = upscaledLR[row-gradientmargin:row+gradientmargin+1, col-gradientmargin:col+gradientmargin+1]
            # Calculate hashkey
            angle, strength, coherence, theta, lamda, u = hashkey(gradientblock, Qangle, weighting)
            if exQ:
                patch = np.concatenate((patch,np.array([theta,lamda,u,1.])),axis=None)
            # Get pixel type
            pixeltype = ((row-margin) % R) * R + ((col-margin) % R)
            predictHR[row-margin,col-margin] = patch.dot(h[angle,strength,coherence,pixeltype])
            if args.groundTruth:
                pixelerror=ycrcv[row,col,0].astype('float')/255.-predictHR[row-margin,col-margin].astype('float')
                classCount[angle,strength,coherence,pixeltype]+=1
                classError[angle,strength,coherence,pixeltype]+=pixelerror*pixelerror

    # Scale back to [0,255]
    predictHR = np.clip(predictHR.astype('float') * 255., 0., 255.)
    # Bilinear interpolation on CbCr field
    result = np.zeros((heightHR, widthHR, 3))

    y = ycrcvorigin[:,:,0]
    bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, y, kind='linear')
    result[:,:,0] = bilinearinterp(widthgridHR, heightgridHR)
    
    cr = ycrcvorigin[:,:,1]
    bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, cr, kind='linear')
    result[:,:,1] = bilinearinterp(widthgridHR, heightgridHR)

    cv = ycrcvorigin[:,:,2]
    bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, cv, kind='linear')
    result[:,:,2] = bilinearinterp(widthgridHR, heightgridHR)

    result[margin:heightHR-margin,margin:widthHR-margin,0] = predictHR
    result = cv2.cvtColor(np.uint8(result), cv2.COLOR_YCrCb2RGB)
    try:
        os.mkdir('results/'+ args.output)
    except Exception as e:
        pass#print("\nignoring error:",e)

    cv2.imwrite('results/'+ args.output + '/' + os.path.splitext(os.path.basename(image))[0] + '.bmp',
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

print('\r', end='')
print(' ' * 60, end='')
print('\rFinished.')
