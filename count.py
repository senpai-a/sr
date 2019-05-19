import cv2
import numpy as np
import os
import sys
import argparse
from math import floor, pi
from matplotlib import pyplot as plt
from scipy import interpolate
from skimage import transform

parser_ = argparse.ArgumentParser()
parser_.add_argument("-i", "--input", help="Specify training set")
args = parser_.parse_args()
#print(args)

# Define parameters
R = 2 #pixel type 1D
patchsize = 11
gradientsize = 9
Qangle = 24
Qstrength = 3
Qcoherence = 3
trainpath = 'train'
if args.input:
    trainpath=args.input

filterSize=patchsize*patchsize

# Calculate the margin
maxblocksize = max(patchsize, gradientsize)
margin = floor(maxblocksize/2)
patchmargin = floor(patchsize/2)
gradientmargin = floor(gradientsize/2)

count=0
# Get image list
imagelist = []
for parent, dirnames, filenames in os.walk(trainpath):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            imagelist.append(os.path.join(parent, filename))


imagecount = 1
for image in imagelist:
    origin = cv2.imread(image)
    height, width,ch = origin.shape
    if height%2==1:
        height-=1
    if width%2==1:
        width-=1
    count+=(height-margin)*(width-margin)
    imagecount += 1

print('sampleN =',count)