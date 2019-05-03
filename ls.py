import numpy as np

def ls2(A, b, λ):
    d=A.shape[1]
    return (b.T.dot(A).dot((np.linalg.inv(A.T.dot(A)+λ*np.eye(d))).T)).T

def ls(ATA,ATb,λ):
    d=ATA.shape[1]
    return np.linalg.inv(ATA+λ*np.eye(d)).dot(ATb)
