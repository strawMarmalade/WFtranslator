import numpy as np
import scipy.linalg as la


def mu(x,q):
    #p. 29
    dist = np.linalg.norm(x-q)
    return np.exp((dist-1/3)**(-1))/(np.exp((dist-1/3)**(-1)) + np.exp((1/2-dist)**(-1)))

def projector(x, x0, A):

    """
    p. 28
    if QR = A is the QR decomp of A
    then I think QQ^T is the ortho. projec. onto the space spanned by A
    
    (x0,a_1,...,a_n) describes the n-dimensional affine space through x0 where 
    a_i are the columns of A. So A is describes the n-dim. linear
    subspace, so that we project onto that and then add x0 back into it
    """
    Q, _ = la.qr(A)
    return Q@Q.T@x+x0

def phi(x, x0, A, q):
    #p. 30
    m = mu(x, q)
    return m*projector(x,x0,A)+(1-m)*x

def findDisc(X):
    return