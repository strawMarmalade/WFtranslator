import numpy as np
import scipy.linalg as la
import time
import networkx as nx
import os

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

def calcDistFromBall(x0,x, fs):
    #Y is a list of points x1...xm
    if len(fs) != 0:
        maxYval = np.max([fs[j](x) for j in range(len(fs))])
        return np.max([np.abs(1-np.linalg.norm(x0-x)), maxYval])
    return np.abs(1-np.linalg.norm(x0-x))

def GHDist(n, x0, X1):
    #p. 23
    outputList = []
    fs = []
    for j in range(n):
        distanceIndex = np.argmin([calcDistFromBall(x0, x, outputList) for x in X1])
        outputList.append(X[distanceIndex])
        fs.append(lambda y: 1/2*(np.linalg.norm(y-x0)**2 - np.linalg.norm(y-outputList[-1])**2 + np.linalg.norm(x0-outputList[-1])**2))
    P = lambda y: y if np.linalg.norm(y)<=1 else y/np.linalg.norm(y)
    F = lambda y: P(np.array([fs[j](y) for j in range(len(fs))]))
    return F

def phi(x, x0, A, q):
    #p. 30
    m = mu(x, q)
    return m*projector(x,x0,A)+(1-m)*x

def calcDistFrom1(x0,x, Y):
    #Y is a list of points x1...xm
    if len(Y) != 0:
        maxYval = np.max([np.linalg.norm(np.dot(x,y)/np.linalg.norm(x)) for y in Y])
        return np.max([np.abs(1-np.linalg.norm(x0-x)), maxYval])
    return np.abs(1-np.linalg.norm(x0-x))

#this already looks very expensive
def findDisc(x0, X, n):
    #p. 26
    outputList = []
    for j in range(n):
        distanceIndex = np.argmin([calcDistFrom1(x0, x, outputList) for x in X])
        outputList.append(X[distanceIndex])
    return outputList

#this pulls each vector in X towards the center of X
#by length 1/r, I hope this is what they mean by rescale
def rescale(X, r):
    m, n = np.shape(X)
    maxEachDim = [np.max([X[:,j]]) for j in range(n)]
    minEachDim = [np.min([X[:,j]]) for j in range(n)]
    center = np.array([maxEachDim[j]-minEachDim[j] for j in range(n)])
    return [x/r+ (1-1/r)*center for x in X]

def fFromPhis(y0, phis):
    y = y0
    for phi in phis:
        y = phi(y)
    return y

def unitBallAroundxinX(x0,X):
    X1 = []
    for x in X:
        if np.linalg.norm(x0-x) <= 1:
            X1.append(x)
    return X1

def kPointsIndeltaNeighOfx(k,delta,x):
    dim = len(x)
    knonfulfilled = k
    insidePoints = []
    while knonfulfilled != 0:
        print(knonfulfilled)
        points = np.reshape(np.random.uniform(0,delta,knonfulfilled*dim), (knonfulfilled,dim))
        for j in range(knonfulfilled):
            if np.linalg.norm(points[j]) <= delta:
                insidePoints.append(x+points[j])
                knonfulfilled -= 1
    return insidePoints


def submanifoldInterpolation(n, r, X):
    print(f"X is of size {len(X):d}, n is {n:d}")
    #p. 62
    #step 1.
    tic = time.perf_counter()
    #X = rescale(X, r)
    X = np.array([X[j]/r for j in range(len(X))])
    toc = time.perf_counter()
    print(f"Rescaling took {toc - tic:0.4f} seconds")
    #step 2.
    #find maximally 1/100-distance set
    s,t = np.shape(X)
    #since this is a symmetric property first create a triangular matrix
    count = 0
    for j in range(s):
        if X[j][0] != -1:
            for i in range(j):
                if np.linalg.norm(X[i]-X[j]) <= 1/100:
                    X[i][0] = -1
    Y = [X[j] for j in range(s) if X[j][0] != -1]
    print(f"Y is of size {len(Y)}")
    tic = time.perf_counter()
    with open('./output6.txt', 'a') as f1:
        f1.write("%%MatrixMarket matrix coordinate pattern symmetric" + os.linesep) 
        for j in range(len(Y)):
            for i in range(j):
                if np.linalg.norm(Y[i]-Y[j]) >= 1/100:
                    count += 1
                    f1.write(f"{i} {j}" + os.linesep)    
    # G = nx.Graph(graph)
    # print(f"Graph has {len(G.edges())} many edges")
    print(count)
    toc = time.perf_counter()
    print(f"Building graph took {toc - tic:0.4f} seconds")
    # tic = time.perf_counter()
    # max_clique = list(nx.approximation.max_clique(G))

def submanifoldInterpolationOld(n, r, X):
    print(f"X is of size {len(X):d}, n is {n:d}")
    #p. 62
    #step 1.
    tic = time.perf_counter()
    #X = rescale(X, r)
    X = np.array([X[j]/r for j in range(len(X))])
    toc = time.perf_counter()
    print(f"Rescaling took {toc - tic:0.4f} seconds")
    #step 2.
    #find maximally 1/100-distance set
    s,t = np.shape(X)
    #since this is a symmetric property first create a triangular matrix
    count = 0
    tic = time.perf_counter()
    with open('./output7.txt', 'a') as f1:
        f1.write("%%MatrixMarket matrix coordinate pattern symmetric" + os.linesep) 
        for j in range(s):
            for i in range(j):
                if np.linalg.norm(X[i]-X[j]) >= 1/100:
                    count += 1
                    f1.write(f"{i} {j}" + os.linesep)    
    # G = nx.Graph(graph)
    # print(f"Graph has {len(G.edges())} many edges")
    print(count)
    toc = time.perf_counter()
    print(f"Building graph took {toc - tic:0.4f} seconds")
    # tic = time.perf_counter()
    # max_clique = list(nx.approximation.max_clique(G))



def gettingF(max_clique, X, n):
    points_q = [X[j] for j in max_clique]
    tic = time.perf_counter()
    X1s = [unitBallAroundxinX(q,X) for q in points_q]
    toc = time.perf_counter()
    print(f"Finding unit ball for qs took {toc - tic:0.4f} seconds")
    tic = time.perf_counter()
    Aq = [np.array(findDisc(points_q[j], X1s[j], n)) for j in range(len(points_q))]
    toc = time.perf_counter()
    print(f"Finding affine space unit disc for qs took {toc - tic:0.4f} seconds")
    tic = time.perf_counter()
    Qs = [la.qr(A)[0] for A in Aq]
    Ps = [lambda y: Qs[val]@Qs[val].T@y+points_q[val] for val in range(len(points_q))]
    toc = time.perf_counter()
    print(f"Finding orthogonal projectors for qs took {toc - tic:0.4f} seconds")
    tic = time.perf_counter()
    mus = [lambda y: mu(y,q) for q in points_q]
    phis = [lambda y: mus[val](y)*Ps[val](y)+(1-mus[val](y))*y for val in range(len(points_q))]
    f = lambda y: fFromPhis(y, phis)
    toc = time.perf_counter()
    print(f"Defining f took {toc - tic:0.4f} seconds\n")
    return f

def kPointsAroundEachxinXwithf(X, f, k, delta):
    points = np.reshape([kPointsIndeltaNeighOfx(k, delta, x) for x in X], (k*len(X),1))
    fs = [f(p) for p in points]
    return (points, fs)
        
X = np.reshape(np.random.uniform(0,1,6000), (1000,6))
x0 = np.random.uniform(0,1,6)

# tic = time.perf_counter()
# q = findDisc(x0, X, 60)
# toc = time.perf_counter()
# print(f"Submanifold find took {toc - tic:0.4f} seconds\n")
#%%
# tic = time.perf_counter()
# submanifoldInterpolation(4, 50, X)
# toc = time.perf_counter()
# print(f"Submanifold find took {toc - tic:0.4f} seconds\n")

