import numpy as np
from numba import njit, prange

#@njit
def cgls(A, b):
    height, width = A.shape
    x = np.zeros((height))
    x[60]=1
    while(True):
        sumA = A.sum()
        if (sumA < 100):
            break
        (sgn,logdet)=np.linalg.slogdet(A)
        if sgn==0 or logdet<0:
            A = A + np.eye(height, width) * sumA * 0.000000005
        else:
            x = np.linalg.inv(A).dot(b)
            break
    return x
