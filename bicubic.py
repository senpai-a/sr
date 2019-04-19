import argparse
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

def gettestargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="output folder name")
    parser.add_argument("-gt", "--groundTruth", help="Use test images as ground truth (down scale them first)",
    action="store_true")
    args = parser.parse_args()
    return args

args = gettestargs()

# Get image list
imagelist = []
for parent, dirnames, filenames in os.walk('test'):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            imagelist.append(os.path.join(parent, filename))

for image in imagelist:
    origin = cv2.imread(image)
    ycrcv = cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)
    if args.groundTruth:
        height, width = ycrcv[:,:,0].shape        
        ycrcvorigin=np.zeros((floor((height+1)/2),floor((width+1)/2),3))
        ycrcvorigin[:,:,0] = transform.resize(ycrcv[:,:,0], (floor((height+1)/2),floor((width+1)/2)), mode='reflect', anti_aliasing=False)
        ycrcvorigin[:,:,1] = transform.resize(ycrcv[:,:,1], (floor((height+1)/2),floor((width+1)/2)), mode='reflect', anti_aliasing=False)
        ycrcvorigin[:,:,2] = transform.resize(ycrcv[:,:,2], (floor((height+1)/2),floor((width+1)/2)), mode='reflect', anti_aliasing=False)
        ycrcvorigin=np.uint8(np.clip(ycrcvorigin.astype('float')*255.,0.,255.))
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
    interp = interpolate.interp2d(widthgridLR, heightgridLR, cr, kind='linear')
    result[:,:,1] = interp(widthgridHR, heightgridHR)

    cv = ycrcvorigin[:,:,2]
    interp = interpolate.interp2d(widthgridLR, heightgridLR, cv, kind='linear')
    result[:,:,2] = interp(widthgridHR, heightgridHR)

    result = cv2.cvtColor(np.uint8(result), cv2.COLOR_YCrCb2RGB)
    try:
        os.mkdir('results/'+ args.output)
    except Exception as e:
        pass#print("\nignoring error:",e)

    cv2.imwrite('results/'+ args.output + '/' + os.path.splitext(os.path.basename(image))[0] + '.bmp',
     cv2.cvtColor(result, cv2.COLOR_RGB2BGR))