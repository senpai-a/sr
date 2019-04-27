from frft import make_E,disfrft,frft2d
import numpy as np
import cv2
from math import pi
from matplotlib import pyplot as plt

im = cv2.imread("SR_testing_datasets/Set5/butterfly.png")
y = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)[:,:,0]
y = y.astype(float)/255.
spec = frft2d(y,0.6,0.6)
a=np.absolute(spec)
b=(np.angle(spec)+pi)/2/pi
rv = frft2d(spec/a,-.6,-.6)
plt.imshow(np.absolute(rv))
plt.show()