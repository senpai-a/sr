import numpy as np
from math import atan2, floor, pi
from numba import njit, prange


def hashkey(block, Qangle, W):
    # Calculate gradient
    gy, gx = np.gradient(block)
    return _hashkey(gx,gy,Qangle,W)

@njit
def _hashkey(gx,gy,Qangle,W):
    # Transform 2D matrix into 1D array
    gx = gx[1:-1,1:-1].ravel()
    gy = gy[1:-1,1:-1].ravel()

    # SVD calculation
    G = np.vstack((gx,gy)).T
    GTWG = G.T.dot(W).dot(G)
    w, v = np.linalg.eig(GTWG)

    # Make sure V and D contain only real numbers
    w = np.real(w)
    v = np.real(v)

    # Sort w and v according to the descending order of w
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:,idx]

    # Calculate theta
    theta = atan2(v[1,0], v[0,0])
    if theta < 0:
        theta = theta + pi

    # Calculate lamda
    lamda = w[0]

    # Calculate u
    sqrtlamda1 = np.sqrt(w[0])
    sqrtlamda2 = np.sqrt(w[1])
    if sqrtlamda1 + sqrtlamda2 == 0:
        u = 0
    else:
        u = (sqrtlamda1 - sqrtlamda2)/(sqrtlamda1 + sqrtlamda2)

    # Quantize
    angle = floor(theta/pi*Qangle)
    if lamda < 0.0001:
        strength = 0
    elif lamda > 0.001:
        strength = 2
    else:
        strength = 1
    if u < 0.25:
        coherence = 0
    elif u > 0.5:
        coherence = 2
    else:
        coherence = 1

    # Bound the output to the desired ranges
    if angle > Qangle-1:
        angle = Qangle-1
    elif angle < 0:
        angle = 0

    return angle, strength, coherence, theta, lamda, u
