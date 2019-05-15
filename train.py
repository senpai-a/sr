import cv2
import numpy as np
import os
import pickle
import sys
from cgls import cgls
from ls import ls
from filterplot import filterplot
from gaussian2d import gaussian2d
import argparse
from hashkey import hashkey
from math import floor, pi
from matplotlib import pyplot as plt
from scipy import interpolate
from skimage import transform
from bicubic import bicubic2x,bicubic0_5x
from numba import njit, prange
from training import collectQV, collectQVrotflip, rotflipQV, resolvefilters

parser_ = argparse.ArgumentParser()
parser_.add_argument("-e", "--extended", help="Use Extended Linear Mapping", action="store_true")
parser_.add_argument("-q", "--qmatrix", help="Use file as Q matrix")
parser_.add_argument("-v", "--vmatrix", help="Use file as V matrix")
parser_.add_argument("-i", "--input", help="Specify training set")
parser_.add_argument("-o", "--output", help="File to save filter")
parser_.add_argument("-p", "--plot", help="Plot the learned filters", action="store_true")
parser_.add_argument("-li", "--linear", help="Use bilinear for init",action="store_true")
parser_.add_argument("-ls", "--ls", help="Use normalized least square with normalization factor lambda -l",action="store_true")
parser_.add_argument("-l", "--l", help="Normalization factor lambda")
parser_.add_argument("-ex2", "--ex2", help="Use normalized features for ExLM",action="store_true")
parser_.add_argument("-cv2", "--cv2", help="Use cv2 interpolation",action="store_true")
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

exQ=args.extended
filterSize=patchsize*patchsize
if exQ:
    filterSize = filterSize + 4
    print("Training Extended Linear Mappings\n")

# Calculate the margin
maxblocksize = max(patchsize, gradientsize)
margin = floor(maxblocksize/2)
patchmargin = floor(patchsize/2)
gradientmargin = floor(gradientsize/2)

#argmin||Qh-V||
Q = np.zeros((Qangle, Qstrength, Qcoherence, R*R, filterSize, filterSize))
V = np.zeros((Qangle, Qstrength, Qcoherence, R*R, filterSize))
h = np.zeros((Qangle, Qstrength, Qcoherence, R*R, filterSize))

classCount = np.zeros((Qangle, Qstrength, Qcoherence, R*R))
coStCount = np.zeros((1001,10000))#coherence 0-1 strength 0-0.1

# Read Q,V from file
if args.qmatrix:
    with open(args.qmatrix, "rb") as fp:
        Q = pickle.load(fp)
if args.vmatrix:
    with open(args.vmatrix, "rb") as fp:
        V = pickle.load(fp)

# Matrix preprocessing
# Preprocessing normalized Gaussian matrix W for hashkey calculation
weighting = gaussian2d([gradientsize, gradientsize], 2)
weighting = np.diag(weighting.ravel())

# Preprocessing permutation matrices P for nearly-free 8x more learning examples
P = np.zeros((patchsize*patchsize, patchsize*patchsize, 7))
rotate = np.zeros((patchsize*patchsize, patchsize*patchsize))
flip = np.zeros((patchsize*patchsize, patchsize*patchsize))

for i in range(0, patchsize*patchsize):
    i1 = i % patchsize
    i2 = floor(i / patchsize)
    j = patchsize * patchsize - patchsize + i2 - patchsize * i1
    rotate[j,i] = 1
    k = patchsize * (i2 + 1) - i1 - 1
    flip[k,i] = 1
#get transfrom matrices P
for i in range(1, 8):
    i1 = i % 4
    i2 = floor(i / 4)
    P[:,:,i-1] = np.linalg.matrix_power(flip,i2).dot(np.linalg.matrix_power(rotate,i1))

# Get image list
imagelist = []
for parent, dirnames, filenames in os.walk(trainpath):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            imagelist.append(os.path.join(parent, filename))

# Compute Q and V
imagecount = 1
for image in imagelist:
    print('\r', end='')
    print(' ' * 60, end='')
    print('\rProcessing image ' + str(imagecount) + ' of ' + str(len(imagelist)) + ' (' + image + ')')
    sys.stdout.flush()

    origin = cv2.imread(image)
    # Extract only the luminance in YCbCr
    grayorigin = cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)[:,:,0]
    # Normalized to [0,1]
    grayorigin = cv2.normalize(grayorigin.astype('float'), None, grayorigin.min()/255, grayorigin.max()/255, cv2.NORM_MINMAX)
    # Downscale (bicubic interpolation)
    height, width = grayorigin.shape
    if height%2==1:
        height-=1
    if width%2==1:
        width-=1
    grayorigin=grayorigin[0:height,0:width]
    LR = bicubic0_5x(grayorigin)
    # Upscale (bilinear interpolation)
    upscaledLR = bicubic2x(LR)
    # Calculate A'A, A'b and push them into Q, V
    if exQ:
        collectQVrotflip(upscaledLR,grayorigin,patchsize,Q,V,weighting,P)
    else:
        collectQV(upscaledLR,grayorigin,patchsize,Q,V,weighting)
    imagecount += 1

if not exQ:
    rotflipQV(Q,V,P)

of="filter.p"
if args.output:
    of=args.output
# Write Q,V to file
with open('q_'+of, "wb") as fp:
    pickle.dump(Q, fp)
with open("v_"+of, "wb") as fp:
    pickle.dump(V, fp)

# Compute filter h
print('\nresolving filters ...')
sys.stdout.flush()
resolvefilters(Q,V,h)

# Write filter to file

with open(of, "wb") as fp:
    pickle.dump(h, fp)

# Plot the learned filters
if args.plot:
    filterplot(h, R, Qangle, Qstrength, Qcoherence, patchsize)

print('\r', end='')
print(' ' * 60, end='')
print('\rFinished.')
