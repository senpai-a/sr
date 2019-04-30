import argparse
import cv2
import numpy as np
import os
import pickle
import sys
from gaussian2d import gaussian2d
from hashkey import hashkey
from math import floor, pi
from matplotlib import pyplot as plt
from scipy import interpolate
from skimage import transform

def bicubic2x(mat):
    mch=False
    if len(mat.shape)==3:
        ch=mat.shape[2]
        mch=True
    if mch:
        h,w,ch=mat.shape
    else:
        h,w=mat.shape
    heightgridLR = np.linspace(0,h-1,h)
    widthgridLR = np.linspace(0,w-1,w)

    heightgridHR = np.linspace(0,h-0.5,h*2)
    widthgridHR = np.linspace(0,w-0.5,w*2)
    heightHR=len(heightgridHR)
    widthHR=len(widthgridHR)
    if mch:
        result = np.zeros((heightHR,widthHR,ch))
    else:
        result = np.zeros((heightHR,widthHR))
    if not mch:        
        interp=interpolate.interp2d(widthgridLR, heightgridLR, mat, kind='cubic')
        result=interp(widthgridHR,heightgridHR)
    else:
        for i in range(ch):
            interp=interpolate.interp2d(widthgridLR, heightgridLR, mat[:,:,i], kind='cubic')
            result[:,:,i]=interp(widthgridHR,heightgridHR)
    result=np.clip(result.astype('float'),0.,255.)
    return result
'''
from getbicubicargs import getbicubicargs
args=getbicubicargs()
# Get image list
tpath = 'test'
if args.input:
    tpath = args.input
imagelist = []
for parent, dirnames, filenames in os.walk(tpath):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            imagelist.append(os.path.join(parent, filename))

for image in imagelist:
    origin = cv2.imread(image)
    ycrcv = cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)
    #downscale
    if args.groundTruth:
        height, width = ycrcv[:,:,0].shape        
        if height%2==1:
            height-=1
        if width%2==1:
            width-=1
        ycrcv = ycrcv[0:height,0:width,:]   
        ycrcvorigin=np.zeros((int(height/2),int(width/2),3))
        ycrcvorigin=cv2.resize(ycrcv,(int(width/2),int(height/2)),interpolation=cv2.INTER_CUBIC)
    else:
        ycrcvorigin=cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)
       
    # Upscale
   
    heightLR, widthLR = ycrcvorigin[:,:,0].shape    
    heightgridLR = np.linspace(0,heightLR-1,heightLR)
    widthgridLR = np.linspace(0,widthLR-1,widthLR)

    heightgridHR = np.linspace(0,heightLR-0.5,heightLR*2)
    widthgridHR = np.linspace(0,widthLR-0.5,widthLR*2)

    heightHR=len(heightgridHR)
    widthHR=len(widthgridHR)

    result = np.zeros((heightHR, widthHR, 3))

    y = ycrcvorigin[:,:,0]
    interp = interpolate.interp2d(widthgridLR, heightgridLR, y, kind='cubic')
    result[:,:,0] = interp(widthgridHR, heightgridHR)

    cr = ycrcvorigin[:,:,1]
    interp = interpolate.interp2d(widthgridLR, heightgridLR, cr, kind='cubic')
    result[:,:,1] = interp(widthgridHR, heightgridHR)

    cv = ycrcvorigin[:,:,2]
    interp = interpolate.interp2d(widthgridLR, heightgridLR, cv, kind='cubic')
    result[:,:,2] = interp(widthgridHR, heightgridHR)

    result=np.clip(result.astype('float'),0.,255.)
    result = cv2.cvtColor(np.uint8(result), cv2.COLOR_YCrCb2RGB)
    
    heightLR, widthLR = ycrcvorigin[:,:,0].shape
    result = np.zeros((heightLR*2, widthLR*2, 3))
    result = cv2.resize(ycrcvorigin,(widthLR*2, heightLR*2),interpolation=cv2.INTER_CUBIC)
    result = cv2.cvtColor(result, cv2.COLOR_YCrCb2RGB)
    
    try:
        os.mkdir('results/'+ args.output)
    except Exception as e:
        pass#print("\nignoring error:",e)

    cv2.imwrite('results/'+ args.output + '/' + os.path.splitext(os.path.basename(image))[0] + '.png',
     cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
     '''