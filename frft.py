import numpy as np

def make_E(N,p):
    d2 = np.matirx([1,-2,1])
    d_p = 1
    s = 0
    st = np.zeros((1,N))
    for k in range(1:int(p/2)+1):
        d_p = np.convolve(d2,d_p)
        st[N-k:N]=d_p[0:k]
        st[0:k+1]=d_p[k:]
        st[0]=0
        temp=np.matrix([range(1,k+1),range(1,k+1)])
        temp=np.ravel(temp.T)/range(1,2*k+1)
        s = s + np.power(-1,k-1)*np.prod(temp)*2*st
    pass

def get_E(N,p):
    pass

#f:离散信号 a:阶数 p:近似阶数，默认len(f)/2
def disfrft(f,a,p=-1):
    N=len(f)
    if p==-1:
        p=len(f)/2 
    if N%2==0:
        even = 1
    else:
        even = 0
    shft = (np.matrix(range(0,N))+int(N/2))%N+1
    f = np.matrix(f).T
    p = min(max(2,p),N-1)
    E = dFRFT(N,p)
    ret = np.zeros(N)
    pass
