import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

def hpf(mat,d0=30):
    spec =fftpack.fftshift(fftpack.fft2(mat))
    order=2
    M,N=mat.shape
    m=M//2
    n=N//2
    ret = np.zeros(mat.shape).astype(complex)
    for i in range(M):
        for j in range(N):
            d=np.sqrt((i-m+1)**2+(j-n+1)**2)
            if d==0:
                h=0
            else:
                h=1/(1+0.414*((d0/d)**(2*order)))
            ret[i,j]=h*spec[i,j]
    ret = fftpack.ifft2(fftpack.ifftshift(ret))
    ret = np.uint8(np.clip(np.real(ret),0.,255.))
    return ret
'''
import cv2
im=cv2.imread('test/zebra.png')
im=cv2.cvtColor(im,cv2.COLOR_BGR2YCrCb)
hp=hpf(im[:,:,0])
plt.imshow(hp)
plt.show()
'''