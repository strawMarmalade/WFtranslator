##%%
import numpy as np
import scipy.linalg as la
import time
import networkx as nx
import os
import jax
import jax.numpy as jnp
from jax import lax
import odl
#from ctypes import *
#import ctypes

def mu(x,q):
    #p. 29
    dist = jnp.linalg.norm(x-q)
    return jnp.exp((dist-1/3)**(-1))/(jnp.exp((dist-1/3)**(-1)) + jnp.exp((1/2-dist)**(-1)))

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
    jax.device_put(Q)
    return Q@Q.T@x+x0

def calcDistFromBall(x0, x, fs):
    #Y is a list of points x1...xm
    if len(fs) != 0:
        maxYval = jnp.max([fs[j](x) for j in jnp.arange(len(fs))])
        return jnp.max([jnp.abs(1-jnp.linalg.norm(x0-x)), maxYval])
    return jnp.abs(1-jnp.linalg.norm(x0-x))

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

def getMaxSet(X, mode):
    Y = []
    X = list(X)
    if mode == "random":
        lenX = len(X)
        while lenX > 0:
            Y.append(X[0])
            j = 0
            while j < lenX:
                if np.linalg.norm(X[j]-Y[-1]) <= 1/100:
                    X.pop(j)
                    #X = np.delete(X,(j), axis=0)
                    lenX -= 1
                    j -= 1
                j += 1
    print(f"Y len is {len(Y)}")     
    return Y

def exportMatOfDist(X):
    count = 0
    with open('./output10shapes.txt', 'a') as f1:
        f1.write("%%MatrixMarket matrix coordinate pattern symmetric" + os.linesep) 
        for j in range(len(X)):
            for i in range(j):
                if np.linalg.norm(X[i]-X[j]) > 1/100:
                    count += 1
                    f1.write(f"{i} {j}" + os.linesep)   
    print(count)

# def pmc(nnodes, nedges, startEdges, endEdges, cliqueArray):
#     #_libpmc = ctypes.CDLL("/home/leo/Documents/work/WFtranslator/pmcdev/build/out/libpmc.so")
#     _libpmc = ctypes.CDLL("/usr/local/lib/libpmc.so")
#     """
#        int res = max_clique(long long nedges, int *ei, int *ej,
#                 int outsize, int *clique,
#                 bool verbose=true,
#                 int algorithm=0,
#                 int threads=1,
#                 bool graph_stats=false,
#                 string heuristic_strategy="kcore",
#                 double time_limit_secs=60*60,
#                 double remove_time_secs = 4.0,
#                 string edge_sorter = "",
#                 string vertex_search_order = "deg",
#                 bool decreasing_order=false
#                 );
#     """
#     _libpmc.max_clique.argtypes = (ctypes.c_longlong, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_bool, ctypes.c_int,ctypes.c_int, ctypes.c_bool,ctypes.POINTER(ctypes.c_char),ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_bool)
#     edgesLen = ctypes.c_longlong(nedges)
#     array_type = ctypes.c_int*nedges
#     array_node_type = ctypes.c_int*nnodes
#     result =  _libpmc.max_clique(edgesLen, array_type(*startEdges), array_type(*endEdges), ctypes.c_int(nnodes), array_node_type(*cliqueArray), ctypes.c_bool(True), ctypes.c_int(0), ctypes.c_int(16), ctypes.c_bool(True), "deg".encode('utf-8'), ctypes.c_double(60*60), ctypes.c_double(4), "".encode('utf-8'), "deg".encode('utf-8'), ctypes.c_bool(False))
#     # _libpmc.print_max_clique.argtypes = (ctypes.c_longlong, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int))
#     # edgesLen = ctypes.c_longlong(nedges)
#     # array_type = ctypes.c_int*nedges
#     # array_node_type = ctypes.c_int*nnodes
#     # result =  _libpmc.print_max_clique(edgesLen, array_type(*startEdges), array_type(*endEdges), ctypes.c_int(nnodes), array_node_type(*cliqueArray))
#     print(result)

    # count = 0
    # fromList = []
    # toList = []
    # for j in range(len(X)):
    #     for i in range(j):
    #         if np.linalg.norm(X[i]-X[j]) > 1/100:
    #             count += 1
    #             fromList.append(j)
    #             toList.append(i)
    # cliqueArray = []*s
    # print("Before pmc")
    # pmc(s, count, fromList, toList, cliqueArray)

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
    tic = time.perf_counter()
    exportMatOfDist(X)
    toc = time.perf_counter()
    print(f"Outputting matrix took {toc - tic:0.4f} seconds")
    tic = time.perf_counter()
    Y = getMaxSet(X, "random")
    toc = time.perf_counter()
    print(f"Get max set took {toc - tic:0.4f} seconds")

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


points = np.reshape(np.fromfile('Heu2.txt', dtype=int, count=-1, sep=' '), (-1, 6))
max_c = [654, 622, 606, 550, 481, 293, 199, 82, 183, 90, 155, 490, 110, 994, 120, 121, 990, 128, 129, 137, 138, 144, 145, 151, 102, 156, 161, 165, 169, 170, 93, 175, 179, 92, 184, 1014, 191, 192, 195, 85, 81, 200, 1018, 875, 218, 219, 76, 222, 225, 845, 232, 841, 73, 242, 243, 72, 1037, 253, 256, 262, 263, 1049, 68, 270, 271, 276, 277, 281, 1059, 285, 290, 294, 64, 299, 1168, 1222, 1355, 658, 655, 546, 649, 648, 646, 640, 1317, 636, 1307, 1303, 623, 618, 617, 612, 611, 605, 602, 599, 598, 595, 594, 585, 581, 580, 577, 576, 573, 572, 567, 566, 560, 1377, 1373, 549, 556, 107, 1226, 422, 423, 426, 427, 542, 538, 535, 534, 438, 441, 442, 531, 446, 447, 450, 451, 454, 455, 530, 460, 461, 466, 467, 472, 473, 476, 477, 480, 54, 484, 487, 48, 527, 491, 526, 496, 497, 502, 503, 504, 505, 510, 511, 519, 520, 1359, 513, 514, 44, 1365, 39, 38, 522, 523, 495, 494, 47, 51, 53, 58, 34, 59, 1369, 435, 434, 433, 541, 432, 543, 61, 1368, 416, 413, 412, 1376, 411, 410, 547, 557, 407, 559, 406, 403, 564, 565, 402, 1227, 1381, 1382, 397, 396, 393, 392, 1230, 1234, 1240, 1242, 584, 1248, 588, 589, 590, 591, 1249, 1252, 1256, 1269, 1260, 1264, 1268, 1272, 603, 1386, 1273, 1387, 1390, 1278, 1282, 615, 1394, 1283, 1285, 1288, 1398, 1289, 1291, 1402, 629, 630, 1294, 632, 1295, 1406, 635, 1299, 1311, 639, 1313, 641, 1321, 1412, 645, 1323, 1324, 1327, 1329, 1420, 1437, 1438, 1335, 1341, 661, 662, 1349, 665, 1351, 668, 669, 670, 671, 674, 675, 676, 321, 679, 680, 320, 319, 1218, 1211, 1210, 1206, 1202, 1198, 1194, 1191, 316, 1188, 1186, 313, 1181, 1180, 1178, 1177, 312, 1174, 1172, 1167, 718, 719, 309, 308, 1163, 1162, 1160, 1159, 239, 1154, 1152, 1151, 305, 304, 1147, 1146, 1143, 1142, 1140, 1361, 1135, 1134, 751, 1128, 756, 757, 1121, 1120, 1103, 1102, 1095, 1091, 1087, 1083, 1080, 1079, 301, 1076, 1075, 1069, 1068, 300, 1065, 1063, 298, 1060, 284, 791, 282, 1055, 1054, 1053, 1052, 799, 278, 67, 265, 1048, 264, 1045, 1044, 809, 1041, 257, 813, 817, 1036, 249, 821, 822, 248, 247, 826, 827, 246, 831, 1032, 833, 1031, 69, 1029, 837, 228, 238, 237, 842, 236, 231, 846, 229, 849, 1025, 1024, 853, 856, 857, 860, 861, 864, 865, 1022, 869, 871, 872, 109, 1021, 226, 215, 214, 211, 208, 881, 884, 885, 887, 888, 207, 1017, 891, 893, 894, 206, 77, 196, 899, 186, 901, 1013, 91, 906, 907, 178, 1010, 911, 1006, 172, 916, 917, 920, 921, 1003, 923, 162, 160, 928, 929, 930, 931, 100, 101, 934, 935, 938, 939, 152, 147, 942, 943, 949, 956, 957, 960, 961, 964, 966, 146, 969, 970, 974, 976, 977, 139, 980, 981, 984, 986, 131, 989, 130, 123, 122, 995, 115, 114, 998, 999, 1000, 362, 983, 946, 985, 936, 948, 971, 937, 973, 991, 950, 993, 975, 951, 952, 963, 953, 965, 944, 945, 900, 924, 922, 1005, 913, 1007, 1008, 1009, 912, 1011, 910, 908, 902, 794, 898, 896, 890, 880, 878, 876, 874, 868, 866, 852, 850, 848, 1028, 838, 1030, 836, 834, 832, 830, 828, 819, 818, 816, 814, 1040, 812, 1042, 1043, 810, 808, 806, 805, 804, 803, 802, 801, 800, 798, 796, 795, 772, 793, 1058, 792, 790, 788, 787, 786, 1064, 785, 1066, 784, 783, 782, 781, 1071, 1072, 1073, 1074, 780, 779, 778, 777, 776, 775, 1081, 774, 773, 1084, 702, 1086, 771, 1088, 770, 1090, 769, 1092, 768, 1094, 767, 1096, 766, 1098, 1099, 1100, 1101, 765, 764, 763, 762, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 761, 760, 759, 758, 1124, 1125, 1126, 1127, 753, 1129, 752, 1131, 1132, 1133, 750, 748, 747, 746, 745, 1139, 744, 1141, 743, 742, 741, 740, 739, 738, 737, 736, 735, 734, 733, 732, 731, 1155, 730, 1157, 729, 728, 727, 726, 725, 724, 723, 722, 721, 720, 715, 714, 713, 1171, 712, 1173, 711, 1175, 710, 709, 708, 707, 706, 705, 704, 703, 647, 1185, 701, 1187, 700, 1189, 699, 698, 697, 696, 695, 1195, 1196, 1197, 694, 1199, 693, 1201, 692, 1203, 691, 1205, 690, 1207, 689, 1209, 688, 687, 686, 685, 1214, 1215, 1216, 1217, 684, 1219, 683, 1221, 682, 1223, 681, 388, 512, 417, 401, 1229, 400, 1231, 391, 1233, 390, 1235, 1236, 1237, 389, 1239, 345, 1241, 387, 1243, 386, 1245, 385, 384, 383, 382, 1250, 1251, 381, 1253, 380, 1255, 379, 1257, 378, 1259, 377, 1261, 376, 1263, 375, 1265, 1266, 1267, 374, 373, 372, 371, 370, 369, 1274, 1275, 368, 1277, 367, 1279, 366, 365, 364, 363, 1284, 0, 361, 360, 359, 358, 1290, 357, 356, 355, 354, 353, 1296, 1297, 1298, 352, 1300, 351, 1302, 350, 1304, 349, 1306, 348, 1308, 347, 1310, 346, 1312, 324, 1314, 344, 1316, 343, 1318, 342, 1320, 341, 340, 339, 338, 337, 1326, 336, 1328, 335, 1330, 334, 1332, 333, 1334, 332, 1336, 1337, 1338, 331, 1340, 330, 1342, 1343, 1344, 1345, 1346, 329, 1348, 328, 1350, 327, 1352, 326, 1354, 325, 1356, 209, 1358, 1, 1360, 45, 1362, 43, 1364, 42, 1366, 1367, 37, 35, 33, 32, 1372, 31, 30, 29, 28, 27, 1378, 26, 1380, 25, 24, 1383, 23, 22, 21, 20, 19, 1389, 18, 1391, 17, 1393, 16, 1395, 15, 1397, 14, 1399, 13, 1401, 12, 1403, 11, 1405, 10, 1407, 1408, 1409, 9, 1411, 8, 1413, 1414, 1415, 1416, 1417, 7, 1419, 6, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 5, 4, 3, 2, 289]

def unitBallAroundxinX(x0,X):
    X1 = []
    for x in X:
        if np.linalg.norm(x0-x) <= 1:
            X1.append(x)
    return X1

# this can all still be put into the rust part
def jaxGettingQs(max_clique, X, n):
    points_q = [X[j] for j in max_clique]
    X1s = [unitBallAroundxinX(q,X) for q in points_q]
    Aq = [np.array(findDisc(points_q[j], X1s[j], n)) for j in range(len(points_q))]
    Qs = [la.qr(A.T)[0] for A in Aq]
    return points_q, Qs

def jaxGettingF(points_q, Qs):
    Ps = [lambda y: Qs[val]@Qs[val].T@y+points_q[val] for val in jnp.arange(len(points_q))]
    mus = [lambda y: mu(y,q) for q in points_q]
    phis = [lambda y: mus[val](y)*Ps[val](y)+(1-mus[val](y))*y for val in jnp.arange(len(points_q))]
    body_fun = lambda i,x: lax.switch(i, phis, x)
    return lambda y: lax.fori_loop(0,len(phis),body_fun,y)

tic = time.perf_counter()
#points_q, Qs = jaxGettingQs(max_c,points, 4)
# Using readline()
points_q = [points[j] for j in max_c]

file1 = open('feffer/QQTvals2.txt', 'r')
Qs = []
while True:  
    # Get next line from file
    line = file1.readline()
  
    # if line is empty
    # end of file is reached
    if not line:
        break
    Qs.append(np.reshape(np.array(np.matrix(line)), (6,6)))

#Qs2 = np.reshape(np.fromfile("feffer/QQTvals2.txt"), (949,6,6))
toc = time.perf_counter()
print(f"it took {toc-tic:04f} seconds to do")
print(Qs[0]@Qs[0].T)
jax.device_put(points_q)
jax.device_put(Qs)
jax.device_put(points)
print("Done with setup")
func = jax.jit(jaxGettingF(points_q, Qs))
grad = jax.jit(jax.grad(func))
print("Done with func calculation")
##%%
#jax.debug.print(grad(points[10]))
x_small = points[10]
print(grad(x_small))


# delta = 0.01
# domainSets = [odl.set.domain.IntervalProd(x-delta,x+delta) for x in points]
# domain = odl.set.sets.SetUnion(domainSets)

# rangeSpace = odl.rn(6, dtype='float32')

# class myOperator(odl.Operator):
#     def __init__(self, func, grad):
#         self.func = func
#         self.is_linear = False
#         self.is_functional = True
#         self.derivative = grad
#         dom = domain
#         ran = rangeSpace
#         super(myOperator, self).__init__(dom, ran)
#     def _call(self, x, out):
#         out = self.func(x)
#     def _call(self, x):
#         return self.func(x)
#     @property
#     def derivative(self, point):
#         return super().derivative(point)

# Op = myOperator(func, grad)


# def gettingF(max_clique, X, n):
#     points_q = [X[j] for j in max_clique]
#     tic = time.perf_counter()
#     X1s = [unitBallAroundxinX(q,X) for q in points_q]
#     toc = time.perf_counter()
#     print(f"Finding unit ball for qs took {toc - tic:0.4f} seconds")
#     tic = time.perf_counter()
#     Aq = [np.array(findDisc(points_q[j], X1s[j], n)) for j in range(len(points_q))]
#     toc = time.perf_counter()
#     print(f"Finding affine space unit disc for qs took {toc - tic:0.4f} seconds")
#     tic = time.perf_counter()
#     Qs = [la.qr(A)[0] for A in Aq]
#     Ps = [lambda y: Qs[val]@Qs[val].T@y+points_q[val] for val in range(len(points_q))]
#     toc = time.perf_counter()
#     print(f"Finding orthogonal projectors for qs took {toc - tic:0.4f} seconds")
#     tic = time.perf_counter()
#     mus = [lambda y: mu(y,q) for q in points_q]
#     phis = [lambda y: mus[val](y)*Ps[val](y)+(1-mus[val](y))*y for val in jnp.arange(len(points_q))]
#     body_fun = lambda i,x: phis[i](x)
#     f = lambda y: lax.fori_loop(0,len(phis),body_fun,y)
#     toc = time.perf_counter()
#     print(f"Defining f took {toc - tic:0.4f} seconds\n")
#     return f

# def kPointsAroundEachxinXwithf(X, f, k, delta):
#     points = np.reshape([kPointsIndeltaNeighOfx(k, delta, x) for x in X], (k*len(X),1))
#     fs = [f(p) for p in points]
#     return (points, fs)
        
# X = np.reshape(np.random.uniform(0,1,6000), (1000,6))
# x0 = np.random.uniform(0,1,6)
# %%
