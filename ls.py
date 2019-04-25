import numpy as np

def ls(A, b, λ):
    d=A.shape[1]
    return (b.T.dot(A).dot((np.linalg.inv(A.T.dot(A)-λ*np.eye(d))).T)).T