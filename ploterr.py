import os
import pickle
import sys
from matplotlib import pyplot as plt

err1 = 'results/filter_91_err/classError.p'
len1 = 'results/filter_91_err/classCount.p'
err2 = 'results/filterEx2_err/classError.p'
len2 = 'results/filterEx2_err/classCount.p'
with open(err1,'rb') as f:
    err1 = pickle.load(f)
with open(len1,'rb') as f:
    len1 = pickle.load(f)
with open(err2,'rb') as f:
    err2 = pickle.load(f)
with open(len2,'rb') as f:
    len2 = pickle.load(f)

mse1 = err1.astype('float')/len1
mse2 = err2.astype('float')/len2
dmse = mse2 - mse1
plt.bar(range(len(dmse)),dmse)
plt.show