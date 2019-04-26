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

def gettestargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="output folder name")
    parser.add_argument("-gt", "--groundTruth", help="Use test images as ground truth (down scale them first)",
    action="store_true")
    parser.add_argument("-i", "--input", help="input folder name")
    args = parser.parse_args()
    return args

args = gettestargs()

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
    result = np.zeros((heightLR*2, widthLR*2, 3))
    result = cv2.resize(ycrcvorigin,(widthLR*2, heightLR*2),interpolation=cv2.INTER_CUBIC)
    result = cv2.cvtColor(result, cv2.COLOR_YCrCb2RGB)    
    try:
        os.mkdir('results/'+ args.output)
    except Exception as e:
        pass#print("\nignoring error:",e)

    cv2.imwrite('results/'+ args.output + '/' + os.path.splitext(os.path.basename(image))[0] + '.png',
     cv2.cvtColor(result, cv2.COLOR_RGB2BGR))