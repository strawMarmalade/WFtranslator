import numpy as np
import matplotlib.pyplot as plt
import warnings
import io
import time
from matplotlib.patches import Ellipse
import math
import torch
#import feffer

"""
For a given gridSize (default 200),
we define the grid of points by
[-1  + 2/(gridSize-1)[n1 
 -1]    n2]                 

so that the index [n1,n2] is the point on the grid
"""

def point2grid(p,gridSize=200):
    return np.round((p+np.array([1,1]))*(gridSize-1)/2).astype(int)

def grid2point(p,gridSize=200):
    return np.array([-1,-1]+2/(gridSize-1)*p)

def rad2ang(angle, angleAcc):
    return np.round(angle/2/np.pi*angleAcc).astype(int)%angleAcc

def ang2rad(angle, angleAcc):
    return angle*2*np.pi/angleAcc
        
def rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], 
                         [np.sin(theta),  np.cos(theta)]])

def genEll(lowEffort = False):
    angle = np.random.uniform(-1,1)*np.pi/4
    
    a, b = np.random.uniform(0,1, 2)
    if lowEffort:
        maxX = np.sqrt((a*np.cos(angle))**2+(b*np.sin(angle))**2)
        maxY = np.sqrt((a*np.sin(angle))**2+(b*np.cos(angle))**2)
        
        #this situation corresponds to the "limits" of the ellipse
        #being outside the unit circle already so we wont shift the center
        if (maxX-maxY)**2 > 1 or (maxX+maxY)**2 > 1:
            radCenter = 0
            center = np.array([0,0])
        else:
            angOfCenter = np.random.uniform(0,1)*2*np.pi
            valPlus = maxX*np.cos(angOfCenter)+maxY*np.sin(angOfCenter)
            valMinus = maxX*np.cos(angOfCenter)-maxY*np.sin(angOfCenter)
            m1 = np.abs(valPlus+np.sqrt(1-(maxX*np.sin(angOfCenter)-maxY*np.cos(angOfCenter))**2))
            m2 = np.abs(-valPlus+np.sqrt(1-(maxX*np.sin(angOfCenter)-maxY*np.cos(angOfCenter))**2))
            m3 = np.abs(valMinus+np.sqrt(1-(maxX*np.sin(angOfCenter)+maxY*np.cos(angOfCenter))**2))
            m4 = np.abs(-valMinus+np.sqrt(1-(maxX*np.sin(angOfCenter)+maxY*np.cos(angOfCenter))**2))
            radCenter = np.random.uniform(0,np.min([m1,m2,m3,m4]))
            center = np.array([radCenter*np.cos(angOfCenter),radCenter*np.sin(angOfCenter)])
    else:
        angOfCenter = np.random.uniform(0,1)*2*np.pi
        t = np.linspace(0,2*np.pi,360)
        Ellrot = np.array([a*np.cos(t)*np.cos(angle)-b*np.sin(t)*np.sin(angle), a*np.cos(t)*np.sin(angle)+b*np.sin(t)*np.cos(angle)])  
        maxs = np.array(Ellrot[0]*np.cos(angOfCenter)+Ellrot[1]*np.sin(angOfCenter)-np.sqrt(1-(Ellrot[0]*np.sin(angOfCenter)-Ellrot[1]*np.cos(angOfCenter))**2))
        centerRad = np.random.uniform(0,np.min(np.abs(maxs)-10e-3))
        center = np.array([centerRad*np.cos(angOfCenter),centerRad*np.sin(angOfCenter)])

    return Ellipse(center, 2*a,2*b, angle=np.rad2deg(angle))

def plotUnitCircle():
    t = np.linspace(0,2*np.pi, 360)
    circ = np.array([np.cos(t), np.sin(t)])
    
    plt.plot(circ[0,:], circ[1,:])


def gridEll(ell, gridSize=200):
    fig = plt.figure(figsize=(1,1), dpi=gridSize, frameon=False)
    ax = fig.add_axes([0,0,1,1])
    ax.add_artist(ell)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.axis('off')
    io_buf = io.BytesIO()
    fig.savefig(io_buf,format='raw', dpi=gridSize)
    io_buf.seek(0)
    grid = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                          newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))[:,:,0] < 255
    io_buf.close()
    plt.close(fig)
    #the reason for the flip and transpose is cuz pyplot has its origin
    #in the top left, and for the grid we have it in the bottom left
    return np.transpose([np.flip(grid[:,val]) for val in range(gridSize)])

def gridFromPolygon(poly, gridSize=200):
    fig = plt.figure(figsize=(1,1), dpi=gridSize, frameon=False)
    ax = fig.add_axes([0,0,1,1])
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.axis('off')
    ax.fill(poly[:,0],poly[:,1], "black")
    io_buf = io.BytesIO()
    fig.savefig(io_buf,format='raw', dpi=gridSize)
    io_buf.seek(0)
    grid = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                          newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))[:,:,0] < 255
    io_buf.close()
    plt.close(fig)
    return np.transpose([np.flip(grid[:,val]) for val in range(gridSize)])

def plotEll(ell, output=True):
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ax.add_artist(ell)
    if output:
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.show()

def generatePolygon(pointNum, offCenter=True, smallestSize=10e-5, niceness=0.1):
    phis = []
    phis.append(np.random.uniform(0, np.pi-niceness))
    for val in range(pointNum):
        new = phis[val]+np.random.uniform(niceness/2, np.pi-niceness)
        if new>2*np.pi:
            new -= 2*np.pi
        phis.append(new)
    phis = np.sort(phis)

    points = []
    rads = np.random.uniform(smallestSize, 1-smallestSize, pointNum)
    points2 = np.array([(rads[val]*np.cos(phis[val]),rads[val]*np.sin(phis[val])) for val in range(pointNum)])
    if offCenter:
        angOfCenter = np.random.uniform(0,1)*2*np.pi
        maxs = np.array(points2[:,0]*np.cos(angOfCenter)+points2[:,1]*np.sin(angOfCenter)-np.sqrt(1-(points2[:,0]*np.sin(angOfCenter)-points2[:,1]*np.cos(angOfCenter))**2))
    
        centerRad = np.random.uniform(0,np.min(np.abs(maxs)))
        
        x0 = centerRad*np.cos(angOfCenter)
        y0 = centerRad*np.sin(angOfCenter)
    else:
        x0 = 0
        y0 =0
    points2 += np.array([x0,y0])

    points = [[x0,y0]]
    points.extend([points2[j] for j in range(pointNum)])
    points.append([x0,y0])
    return np.array(points)

#maybe wont work, i dunno
# def checkIfPointInEllipse(p, ell):
#     center, angle, a, b = ell
#     pRotatedBack = rot(-angle)@(p-center)
#     if -a <= pRotatedBack[0] <= a:
#         if -b*np.sqrt(1-((pRotatedBack[0]/a))**2) <= pRotatedBack[1] <= b*np.sqrt(1-((pRotatedBack[0]/a))**2):
#             return True
#     return False

def drawGrid(grid):
    plt.imshow(grid, origin='lower') 
    plt.show()

def drawPolygon(points, output=True):
    plt.plot(points[:,0], points[:,1])
    if output:
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.show()

def ellipseToWFsetList(ell,gridSize=200, angleAccuracy=360, altMethod=False):
    a = ell.get_width()/2
    b = ell.get_height()/2
    x0,y0 = ell.get_center()
    angle = np.deg2rad(ell.get_angle())

    if altMethod:
        t = np.linspace(-np.pi/2, 3/2*np.pi, angleAccuracy)
        Ellrot = np.array([a*np.cos(t)*np.cos(angle)-b*np.sin(t)*np.sin(angle)+x0, a*np.cos(t)*np.sin(angle)+b*np.sin(t)*np.cos(angle)+y0])  

        angles = [rad2ang(angle+np.arctan2(np.tan(j)*a,b),angleAccuracy) for j in np.linspace(-np.pi/2,np.pi/2, angleAccuracy//2)]
        angles.extend([rad2ang(np.pi+angle+np.arctan(np.tan(j)*a/b),angleAccuracy) for j in np.linspace(np.pi/2, 3/2*np.pi, angleAccuracy//2)])
        angles[angleAccuracy//2] += angleAccuracy//2
        return [[point2grid(np.array([Ellrot[0,j],Ellrot[1,j]]),gridSize=gridSize),[angles[j]]] for j in range(angleAccuracy)]
    t = np.linspace(0, 2*np.pi, angleAccuracy)
    Ellrot = np.array([a*np.cos(t)*np.cos(angle)-b*np.sin(t)*np.sin(angle)+x0, a*np.cos(t)*np.sin(angle)+b*np.sin(t)*np.cos(angle)+y0])  

    angles = [rad2ang(3/2*np.pi+np.arctan2(Ellrot[1,j+1]-Ellrot[1,j],Ellrot[0,j+1]-Ellrot[0,j]),angleAccuracy) for j in range(angleAccuracy-1)]
    angles.extend([rad2ang(3/2*np.pi+np.arctan2(Ellrot[1,-1]-Ellrot[1,-2],Ellrot[0,-1]-Ellrot[0,-2]),angleAccuracy)])
    return [[point2grid(np.array([Ellrot[0,j],Ellrot[1,j]]),gridSize=gridSize),[angles[j]]] for j in range(angleAccuracy)]

    
    # #both of these calculate the rotated ellipse and are almost equally fast...
    # #Ell = np.array([a*np.cos(t), b*np.sin(t)])  
    # # r = rot(angle)
    # # Ellrot = np.zeros((2,Ell.shape[1]))
    # # for i in range(Ell.shape[1]):
    # #     Ellrot[:,i] = r@Ell[:,i]
    # # plt.xlim(-1,1)
    # # plt.ylim(-20,20)
    # # plt.plot(x0+Ell_rot[0,:] , y0+Ell_rot[1,:],'darkorange' ) 
    # #rotated ellipse
    # #plt.plot(Ellrot[0,:],Ellrot[1,:])
    # # WFSetList = [[point2grid(np.array([x0+Ell_rot[0,j],y0+Ell_rot[1,j]])),[np.round((angle+np.arctan2(1,2*b*(a**(-2))*Ell[0,j]/(np.sqrt(1-(Ell[0,j]/a)**2))))*angleAccuracy/(2*np.pi)).astype(int)%angleAccuracy]] for j in range(angleAccuracy)]
    # # x = np.linspace(-a, a, gridSize*2,endpoint=True)
    # # yPlus = [b*np.sqrt(1-a**(-2)*(val**2))+y0 for val in x]
    # # yMinus = [-b*np.sqrt(1-a**(-2)*(val**2))+y0 for val in x]
    # #we allow the below things to divide by zero because
    # #arctan2 can handle when one of the parameters is infinity
    # #but I dont want to have to see the warnings so I supress them


def polygonToWFsetList(poly, gridSize=200, angleAccuracy=360):
    WFSetList = []
    #as the last element of the polygon is the first element, we look at the second last element
    #and take the edge from there to the first vertex
    #towardPointLineMid = 0.5*poly[0]-0.5*poly[-2]
    for val in range(len(poly)-1):
        awayPointLineMid = 0.5*poly[val+1]-0.5*poly[val]
        #arctan2 gives angle between -pi and pi 
        #towardAngle = np.arctan2(towardPointLineMid[1],towardPointLineMid[0])
        awayAngle = np.arctan2(awayPointLineMid[1],awayPointLineMid[0])
        outwardNormalAwayAngle = awayAngle-np.pi/2
        
        outwardWFEndAngle = rad2ang(outwardNormalAwayAngle, angleAccuracy)
        
        #this adds the wavefront outward angle to each intermediate point between
        #the two vertices
        #distBetweenPoints = np.linalg.norm(poly[val]-poly[val+1])
        interPoints = np.round((gridSize-1)/2*np.linalg.norm(poly[val]-poly[val+1])).astype(int)
        #interPoints = interPoints.astype(int)
        
        #the following list will be filled with every angle between the two outward pointing
        #angles from above which is the set of outward wavefront directions for a corner point
        #of a polygon     
        WFSetList.extend([[point2grid(poly[val]+k/interPoints*(poly[val+1]-poly[val]), gridSize=gridSize),[outwardWFEndAngle]] for k in range(interPoints+1)])
        
        pointAsGrid = point2grid(poly[val],gridSize=gridSize)
        
        WFSetList.append([pointAsGrid,list(range(0,angleAccuracy))])
        
        # towardAngleBackward = (angleAccuracy//2+rad2ang(towardAngle,angleAccuracy))%angleAccuracy
        # awayAngleDegrees = rad2ang(awayAngle,angleAccuracy)
        # if towardAngleBackward <= awayAngleDegrees:
        #     WFSetList.append([pointAsGrid,list(range(towardAngleBackward,awayAngleDegrees+1))])
        # else:
        #     vals1 = list(range(towardAngleBackward,angleAccuracy))
        #     vals1.extend(list(range(awayAngleDegrees+1)))
        #     WFSetList.append([pointAsGrid,vals1])

        #when going to next point the one away line will turn into the toward line for the next point
        #towardPointLineMid = awayPointLineMid
    return WFSetList

def drawEllipseBoundary(ell, output=True):
    a = ell.get_width()/2
    b = ell.get_height()/2
    x0,y0 = ell.get_center()
    angle = np.deg2rad(ell.get_angle())
    t = np.linspace(0, 2*np.pi, 360)
    Ellrot = np.array([a*np.cos(t)*np.cos(angle)-b*np.sin(t)*np.sin(angle)+x0, a*np.cos(t)*np.sin(angle)+b*np.sin(t)*np.cos(angle)+y0])
    plt.plot(Ellrot[0,:],Ellrot[1,:])
    if output:
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.show()

def drawWFSetList(WFSetList,gridSize=200, angleAccuracy=360, saveFile=True):
    for val in range(len(WFSetList)):
        pointGrid = WFSetList[val][0]
        point = grid2point(pointGrid,gridSize)
        angles = WFSetList[val][1]
        for angle in angles:
            #to plot the WFset we just make small lines in the correct direction
            #at the point
            angleRad = ang2rad(angle, angleAccuracy)
            vec = [0.05*np.cos(angleRad), 0.05*np.sin(angleRad)]
            plt.plot([point[0],point[0]+vec[0]],[point[1],point[1]+vec[1]],color='black',linewidth=0.3)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    if saveFile:
        plt.savefig('file.png',dpi=300)
    plt.show()

def convertWFListToWFGridLeoConvention(List, gridSize=200, angleAccuracy=360):
    LeoAngleAcc = angleAccuracy//2
    WFSetGrid = np.zeros([gridSize,gridSize, LeoAngleAcc])
    for val in List:
        point = val[0]
        angleListHalf = [ang%LeoAngleAcc for ang in val[1]]
        WFSetGrid[point[0],point[1], angleListHalf] = 1
    return WFSetGrid

def checkIfPointInTriangle(p,xs):
    a = 1/2 * (-xs[1,1] * xs[2,0] + xs[0,1] * (-xs[1,0] + xs[2,0]) + xs[0,0] * (xs[1,1] - xs[2,1]) + xs[1,0] * xs[2,1])
    s = (xs[0,1]*xs[2,0] - xs[0,0]*xs[2,1] + (xs[2,1] - xs[0,1])*p[0] + (xs[0,0] - xs[2,0])*p[1])
    t = (xs[0,0]*xs[1,1] - xs[0,1]*xs[1,0] + (xs[0,1] - xs[1,1])*p[0] + (xs[1,0] - xs[0,0])*p[1])

    return s > 0 and t > 0 and (s + t) < 2 * a

def checkIfPointIsInPolygon(p, poly):
    #as the first and last element in the list poly is the same starting point, we leave it away
    for index in range(1,len(poly)-2):
        xs = np.array([poly[0],poly[index],poly[index+1],poly[0]])
        if np.min(xs[:,0]) <= p[0] <= np.max(xs[:,0]) and np.min(xs[:,1]) <= p[1] <= np.max(xs[:,1]):
        #the only polgons we can construct are star-convex so we just have to check for
        #all the triangles that make up the polygon
            if checkIfPointInTriangle(p, xs):
                return True
    return False


def fullPolygonRoutineTimer(polySize=5, gridSize=200,angleAccuracy=360):
    print(f"Polygon is of size {polySize:d} and grid size is {gridSize:d}")

    tic = time.perf_counter()
    poly = generatePolygon(polySize)
    toc = time.perf_counter()
    print(f"Polygon generation took {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    WFSetList = polygonToWFsetList(poly,gridSize=gridSize,angleAccuracy=angleAccuracy)
    toc = time.perf_counter()
    print(f"Wavefrontset calculation took {toc - tic:0.4f} seconds")
    
    tic = time.perf_counter()
    plotUnitCircle()
    drawPolygon(poly, output=False)
    drawWFSetList(WFSetList, gridSize=gridSize, angleAccuracy=angleAccuracy, saveFile=False)
    toc = time.perf_counter()
    print(f"Plotting polygon with wavefrontset picture took {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    grid = gridFromPolygon(poly, gridSize=gridSize)
    toc = time.perf_counter()
    print(f"Get inside of polygon as grid took {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    drawGrid(grid)
    toc = time.perf_counter()
    print(f"Drawing the grid of polygon took {toc - tic:0.4f} seconds\n")

def fullEllipseRoutineTimer(gridSize = 200, angleAccuracy=360):   
    print(f"Grid size is {gridSize:d}")

    tic = time.perf_counter()
    ell = genEll() 
    toc = time.perf_counter()
    print(f"Low effort ellipse generation took {toc - tic:0.4f} seconds")
    
    tic = time.perf_counter()
    ell = genEll(lowEffort= False) 
    toc = time.perf_counter()
    print(f"High effort ellipse generation took {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    WFSetList = ellipseToWFsetList(ell, gridSize=gridSize, angleAccuracy=angleAccuracy)
    toc = time.perf_counter()
    print(f"Wavefrontset calculation took {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    plotUnitCircle()
    drawEllipseBoundary(ell, output=False)
    drawWFSetList(WFSetList, gridSize=gridSize, angleAccuracy=angleAccuracy, saveFile=False)
    toc = time.perf_counter()
    print(f"Plotting ellipse with wavefrontset picture took {toc - tic:0.4f} seconds")
    
    tic = time.perf_counter()
    grid = gridEll(ell, gridSize=gridSize)
    toc = time.perf_counter()
    print(f"Get inside of ellipse as grid took {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    drawGrid(grid)
    toc = time.perf_counter()
    print(f"Drawing the grid of ellipse took {toc - tic:0.4f} seconds\n")

def canonicalplus1(r, alpha, phi):
    #sometimes there are apparently large enough errors accumulating that it gets outside of -1,1 range,so just get rid of those cases
    val = r*math.cos(alpha - phi)
    if val >= 1:
        return phi
    elif val <= -1:
        print("yes")
        return np.pi + phi
    return (math.acos(val) + phi)
def canonicalminus1(r, alpha, phi):
    val = r*math.cos(alpha - phi)
    if val >= 1:
        return phi
    elif val <= -1:
        return -np.pi + phi
    return (-math.acos(val) + phi)
    #return (-math.acos(r*math.cos(alpha - phi)) + phi)
def canonicalplus2(r, alpha, phi):
    return -math.asin(r*math.cos(alpha-phi)/2)
def canonicalminus2(r, alpha, phi):
    return math.asin(r*math.cos(alpha-phi)/2)
def traveltimeplus(r, alpha, phi):
    return math.sqrt(4 -(r* math.cos(alpha - phi))**2) - r*math.sin(alpha-phi)
def traveltimeminus(r, alpha, phi):
    return math.sqrt(4 -(r* math.cos(alpha - phi))**2) + r*math.sin(alpha-phi)
def pullback(rho, theta, phi, t):
    JMatrix = np.array([[-2* math.sin(rho) + t* math.sin(theta + rho), 2*math.cos(rho) - t* math.cos(theta+rho)],[ t*math.sin(theta+rho), -t*math.cos(theta+rho)]])
    ImageWFVect = np.array([math.cos(phi), math.sin(phi)])
    SinoWFVect = JMatrix.dot(ImageWFVect)
    return round(math.degrees(math.acos(SinoWFVect[0]/math.sqrt(SinoWFVect[0]**2 + SinoWFVect[1]**2))))

def getSinoWF(ImageWF, N=201):
    SinoWF = np.zeros([N,180,180])
    rowindex = 0
    while (rowindex <= N-1):
        colindex = 0
        while (colindex <= N-1):
            WFangleindex = 0
            while (WFangleindex <= 179):
                if ImageWF[rowindex, colindex, WFangleindex] ==1:
                   radius = math.sqrt((2*rowindex/(N-1) -1)**2 + (2*colindex/(N-1) - 1)**2)
                   #computes the distance of the pixel from the origin
                   if radius ==0:
                       positionangle = 0
                   else:
                       #print((2 * colindex / (N - 1) - 1) / radius)
                       #print(math.acos((2 * colindex / (N - 1) - 1) / radius))
                       positionangle = np.sign((2*rowindex/(N-1) -1))*math.acos((2 * colindex / (N - 1) - 1) / radius)
                       if np.sign((2*rowindex/(N-1) -1))== 0:
                           positionangle = math.acos((2 * colindex / (N - 1) - 1) / radius)
                       #print(positionangle)
                   #positionangle is the angle of the position measured in Radians. It takes the range between -pi to pi
                   WFangleradian = math.radians(WFangleindex)
                   #turns WFangle from entry index to radians
                   boundaryradplus = canonicalplus1(radius, positionangle, WFangleradian)
                   #above function returns location on the boundary of circle in radians.
                   # So the range is a float between 0 and 2pi
                   boundarydegreeplus = math.degrees(boundaryradplus)
                   boundaryindexplus = round(boundarydegreeplus *N/360)%N
                   boundaryradminus = canonicalminus1(radius, positionangle, WFangleradian)
                   boundarydegreeminus = math.degrees(boundaryradminus)
                   boundaryindexminus = round(boundarydegreeminus *N/360)%N
                   incomingradplus = canonicalplus2(radius, positionangle, WFangleradian)
                   #above function returns incoming direction in radians relative to the inward pointing normal
                   #so the range is an integer between -pi/2 degrees to pi/2 degrees
                   incomingdegreeplus = math.degrees(incomingradplus)
                   incomingradminus = - incomingradplus
                   incomingdegreeminus = -incomingdegreeplus
                   incomingindexplus = round(incomingdegreeplus + 90)%180
                   incomingindexminus = round(incomingdegreeminus + 90)%180
                   tplus = traveltimeplus(radius, positionangle, WFangleradian)
                   tminus = traveltimeminus(radius, positionangle, WFangleradian)
                   SinoWFindexplus = pullback(boundaryradplus, incomingradplus, WFangleradian, tplus)
                   SinoWFindexminus = pullback(boundaryradminus, incomingradminus, WFangleradian, tminus)
                   SinoWF[boundaryindexplus, incomingindexplus, SinoWFindexplus] = 1
                   SinoWF[boundaryindexminus, incomingindexminus, SinoWFindexminus] = 1
                WFangleindex = WFangleindex + 1
            colindex = colindex + 1
        rowindex = rowindex + 1
    return torch.tensor(SinoWF)

def getSinoWFFromList(WFList, N=201):
    SinoWF = np.zeros([N,180,180])
    for val in WFList:
        pointGrid = val[0]
        angles = [ang%180 for ang in val[1]]
        rowindex = pointGrid[0]
        colindex = pointGrid[1]
        radius = math.sqrt((2*rowindex/(N-1) -1)**2 + (2*colindex/(N-1) - 1)**2)
        #computes the distance of the pixel from the origin
        if radius ==0:
            positionangle = 0
        else:
            #print((2 * colindex / (N - 1) - 1) / radius)
            #print(math.acos((2 * colindex / (N - 1) - 1) / radius))
            positionangle = np.sign((2*rowindex/(N-1) -1))*math.acos((2 * colindex / (N - 1) - 1) / radius)
            if np.sign((2*rowindex/(N-1) -1))== 0:
                positionangle = math.acos((2 * colindex / (N - 1) - 1) / radius)
        for WFangleindex in angles:
            #print(positionangle)
            #positionangle is the angle of the position measured in Radians. It takes the range between -pi to pi
            WFangleradian = math.radians(WFangleindex)
            #turns WFangle from entry index to radians
            boundaryradplus = canonicalplus1(radius, positionangle, WFangleradian)
            #above function returns location on the boundary of circle in radians.
            # So the range is a float between 0 and 2pi
            boundarydegreeplus = math.degrees(boundaryradplus)
            boundaryindexplus = round(boundarydegreeplus *N/360)%N
            boundaryradminus = canonicalminus1(radius, positionangle, WFangleradian)
            boundarydegreeminus = math.degrees(boundaryradminus)
            boundaryindexminus = round(boundarydegreeminus *N/360)%N
            incomingradplus = canonicalplus2(radius, positionangle, WFangleradian)
            #above function returns incoming direction in radians relative to the inward pointing normal
            #so the range is an integer between -pi/2 degrees to pi/2 degrees
            incomingdegreeplus = math.degrees(incomingradplus)
            incomingradminus = - incomingradplus
            incomingdegreeminus = -incomingdegreeplus
            incomingindexplus = round(incomingdegreeplus + 90)%180
            incomingindexminus = round(incomingdegreeminus + 90)%180
            tplus = traveltimeplus(radius, positionangle, WFangleradian)
            tminus = traveltimeminus(radius, positionangle, WFangleradian)
            SinoWFindexplus = pullback(boundaryradplus, incomingradplus, WFangleradian, tplus)%180
            SinoWFindexminus = pullback(boundaryradminus, incomingradminus, WFangleradian, tminus)%180
            SinoWF[boundaryindexplus, incomingindexplus, SinoWFindexplus] = 1
            SinoWF[boundaryindexminus, incomingindexminus, SinoWFindexminus] = 1
    return SinoWF

def dim4getSinoWFFromList(WFList, N=201):
    SinoWF = []
    for val in WFList:
        pointGrid = val[0]
        angles = [ang%180 for ang in val[1]]
        rowindex = pointGrid[0]
        colindex = pointGrid[1]
        radius = math.sqrt((2*rowindex/(N-1) -1)**2 + (2*colindex/(N-1) - 1)**2)
        #computes the distance of the pixel from the origin
        if radius ==0:
            positionangle = 0
        else:
            #print((2 * colindex / (N - 1) - 1) / radius)
            #print(math.acos((2 * colindex / (N - 1) - 1) / radius))
            positionangle = np.sign((2*rowindex/(N-1) -1))*math.acos((2 * colindex / (N - 1) - 1) / radius)
            if np.sign((2*rowindex/(N-1) -1))== 0:
                positionangle = math.acos((2 * colindex / (N - 1) - 1) / radius)
        for WFangleindex in angles:
            #print(positionangle)
            #positionangle is the angle of the position measured in Radians. It takes the range between -pi to pi
            WFangleradian = math.radians(WFangleindex)
            #turns WFangle from entry index to radians
            boundaryradplus = canonicalplus1(radius, positionangle, WFangleradian)
            #above function returns location on the boundary of circle in radians.
            # So the range is a float between 0 and 2pi
            boundarydegreeplus = math.degrees(boundaryradplus)
            boundaryindexplus = round(boundarydegreeplus *N/360)%N
            boundaryradminus = canonicalminus1(radius, positionangle, WFangleradian)
            boundarydegreeminus = math.degrees(boundaryradminus)
            boundaryindexminus = round(boundarydegreeminus *N/360)%N
            incomingradplus = canonicalplus2(radius, positionangle, WFangleradian)
            #above function returns incoming direction in radians relative to the inward pointing normal
            #so the range is an integer between -pi/2 degrees to pi/2 degrees
            incomingdegreeplus = math.degrees(incomingradplus)
            incomingradminus = - incomingradplus
            incomingdegreeminus = -incomingdegreeplus
            incomingindexplus = round(incomingdegreeplus + 90)%180
            incomingindexminus = round(incomingdegreeminus + 90)%180
            tplus = traveltimeplus(radius, positionangle, WFangleradian)
            tminus = traveltimeminus(radius, positionangle, WFangleradian)
            SinoWFindexplus = pullback(boundaryradplus, incomingradplus, WFangleradian, tplus)%180
            SinoWFindexminus = pullback(boundaryradminus, incomingradminus, WFangleradian, tminus)%180
            angPlus = point2grid(np.array([np.sin(np.deg2rad(SinoWFindexplus)), np.cos(np.deg2rad(SinoWFindexplus))]))
            angMinus = point2grid(np.array([np.sin(np.deg2rad(SinoWFindexminus)), np.cos(np.deg2rad(SinoWFindexminus))]))
            SinoWF.append(np.array([boundaryindexplus,incomingindexplus,angPlus[0],angPlus[1]]))
            SinoWF.append(np.array([boundaryindexminus,incomingindexminus,angMinus[0],angMinus[1]]))
    return np.array(SinoWF)

def dim3getSinoWFFromList(WFList, N=201):
    SinoWF = []
    for val in WFList:
        pointGrid = val[0]
        angles = [ang%180 for ang in val[1]]
        rowindex = pointGrid[0]
        colindex = pointGrid[1]
        radius = math.sqrt((2*rowindex/(N-1) -1)**2 + (2*colindex/(N-1) - 1)**2)
        #computes the distance of the pixel from the origin
        if radius ==0:
            positionangle = 0
        else:
            #print((2 * colindex / (N - 1) - 1) / radius)
            #print(math.acos((2 * colindex / (N - 1) - 1) / radius))
            positionangle = np.sign((2*rowindex/(N-1) -1))*math.acos((2 * colindex / (N - 1) - 1) / radius)
            if np.sign((2*rowindex/(N-1) -1))== 0:
                positionangle = math.acos((2 * colindex / (N - 1) - 1) / radius)
        for WFangleindex in angles:
            #print(positionangle)
            #positionangle is the angle of the position measured in Radians. It takes the range between -pi to pi
            WFangleradian = math.radians(WFangleindex)
            #turns WFangle from entry index to radians
            boundaryradplus = canonicalplus1(radius, positionangle, WFangleradian)
            #above function returns location on the boundary of circle in radians.
            # So the range is a float between 0 and 2pi
            boundarydegreeplus = math.degrees(boundaryradplus)
            boundaryindexplus = round(boundarydegreeplus *N/360)%N
            boundaryradminus = canonicalminus1(radius, positionangle, WFangleradian)
            boundarydegreeminus = math.degrees(boundaryradminus)
            boundaryindexminus = round(boundarydegreeminus *N/360)%N
            incomingradplus = canonicalplus2(radius, positionangle, WFangleradian)
            #above function returns incoming direction in radians relative to the inward pointing normal
            #so the range is an integer between -pi/2 degrees to pi/2 degrees
            incomingdegreeplus = math.degrees(incomingradplus)
            incomingradminus = - incomingradplus
            incomingdegreeminus = -incomingdegreeplus
            incomingindexplus = round(incomingdegreeplus + 90)%180
            incomingindexminus = round(incomingdegreeminus + 90)%180
            tplus = traveltimeplus(radius, positionangle, WFangleradian)
            tminus = traveltimeminus(radius, positionangle, WFangleradian)
            SinoWFindexplus = pullback(boundaryradplus, incomingradplus, WFangleradian, tplus)%180
            SinoWFindexminus = pullback(boundaryradminus, incomingradminus, WFangleradian, tminus)%180
            SinoWF.append(np.array([boundaryindexplus,incomingindexplus,SinoWFindexplus]))
            SinoWF.append(np.array([boundaryindexminus,incomingindexminus,SinoWFindexminus]))
    return np.array(SinoWF)

def dim3getSinoWFFromListAsGrid(WFList, N=201):
    SinoWF = np.zeros((N+1, 180, 180), dtype=torch.float16)
    for val in WFList:
        pointGrid = val[0]
        angles = [ang%180 for ang in val[1]]
        rowindex = pointGrid[0]
        colindex = pointGrid[1]
        radius = math.sqrt((2*rowindex/(N-1) -1)**2 + (2*colindex/(N-1) - 1)**2)
        #computes the distance of the pixel from the origin
        if radius ==0:
            positionangle = 0
        else:
            #print((2 * colindex / (N - 1) - 1) / radius)
            #print(math.acos((2 * colindex / (N - 1) - 1) / radius))
            positionangle = np.sign((2*rowindex/(N-1) -1))*math.acos((2 * colindex / (N - 1) - 1) / radius)
            if np.sign((2*rowindex/(N-1) -1))== 0:
                positionangle = math.acos((2 * colindex / (N - 1) - 1) / radius)
        for WFangleindex in angles:
            #print(positionangle)
            #positionangle is the angle of the position measured in Radians. It takes the range between -pi to pi
            WFangleradian = math.radians(WFangleindex)
            #turns WFangle from entry index to radians
            boundaryradplus = canonicalplus1(radius, positionangle, WFangleradian)
            #above function returns location on the boundary of circle in radians.
            # So the range is a float between 0 and 2pi
            boundarydegreeplus = math.degrees(boundaryradplus)
            boundaryindexplus = round(boundarydegreeplus *N/360)%N
            boundaryradminus = canonicalminus1(radius, positionangle, WFangleradian)
            boundarydegreeminus = math.degrees(boundaryradminus)
            boundaryindexminus = round(boundarydegreeminus *N/360)%N
            incomingradplus = canonicalplus2(radius, positionangle, WFangleradian)
            #above function returns incoming direction in radians relative to the inward pointing normal
            #so the range is an integer between -pi/2 degrees to pi/2 degrees
            incomingdegreeplus = math.degrees(incomingradplus)
            incomingradminus = - incomingradplus
            incomingdegreeminus = -incomingdegreeplus
            incomingindexplus = round(incomingdegreeplus + 90)%180
            incomingindexminus = round(incomingdegreeminus + 90)%180
            tplus = traveltimeplus(radius, positionangle, WFangleradian)
            tminus = traveltimeminus(radius, positionangle, WFangleradian)
            SinoWFindexplus = pullback(boundaryradplus, incomingradplus, WFangleradian, tplus)%180
            SinoWFindexminus = pullback(boundaryradminus, incomingradminus, WFangleradian, tminus)%180
            SinoWF[boundaryindexplus,incomingindexplus,SinoWFindexplus] = 1
            SinoWF[boundaryindexminus,incomingindexminus,SinoWFindexminus]  = 1
    return SinoWF

def dim4WFList(WFList):
    WF = []
    for val in WFList:
        pointGrid = val[0]
        x = pointGrid[0]
        y = pointGrid[1]
        angles = [ang%180 for ang in val[1]]
        for angle in angles:
            ang = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
            p = point2grid(ang)
            WF.append(np.array([x,y,p[0],p[1]]))
            ang2 = np.array([np.cos(np.deg2rad(angle+180)), np.sin(np.deg2rad(angle+180))])
            p2 = point2grid(ang2)
            WF.append(np.array([x,y,p2[0],p2[1]]))
    return np.array(WF)

def dim3WFList(WFList):
    WF = []
    for val in WFList:
        pointGrid = val[0]
        x = pointGrid[0]
        y = pointGrid[1]
        angles = [ang%180 for ang in val[1]]
        for angle in angles:
            WF.append(np.array([x,y,angle]))
            WF.append(np.array([x,y,angle+180]))
    return np.array(WF)

def dim3WFListGridNoDouble(WFList, N=201):
    WF = np.zeros((202,202,180), dtype=bool)
    for val in WFList:
        pointGrid = val[0]
        x = pointGrid[0]
        y = pointGrid[1]
        angles = [ang%180 for ang in val[1]]
        for angle in angles:
            WF[x,y,angle] = True
    return WF

def genData(amount, N=201):
    WFDataSinos = []
    WFData = []
    for counter in range(amount):
        print(counter)
        # if counter > amount//2:
        #     randSize = np.random.randint(2, 4)
        #     shape = generatePolygon(randSize)
        #     WFSetList = polygonToWFsetList(shape, gridSize=N, angleAccuracy=360)
        # else:
        shape = genEll()
        WFSetList = ellipseToWFsetList(shape, gridSize=N, angleAccuracy=360)
        WF = dim3WFListGridNoDouble(WFSetList, N=N)
        SinoWF = dim3getSinoWFFromListAsGrid(WFSetList, N=N)
        #arr = [np.array([SinoWF[j][0], SinoWF[j][1], WF[j][0], WF[j][1], SinoWF[j][2], WF[j][2]]) for j in range(len(WF))]
        WFData.append(WF)
        WFDataSinos.append(SinoWF)
    return (WFDataSinos,WFData)

def WFListToPairOfPics(WFSetList, N=201):
    WFSetGrid = torch.tensor(convertWFListToWFGridLeoConvention(WFSetList, gridSize=N, angleAccuracy=360))
    SinoWF = getSinoWFFromList(WFSetList, N=N)
    return (WFSetGrid, SinoWF)

def SheppLogShapeToData(shape, isPoly, N=201):
    if isPoly:
        WFSetList = polygonToWFsetList(shape, gridSize=N, angleAccuracy=360)
        grid = torch.tensor(gridFromPolygon(shape, gridSize=N))
    else:
        WFSetList = ellipseToWFsetList(shape, gridSize=N, angleAccuracy=360)
        grid = torch.tensor(gridEll(shape, gridSize=N))
    return (grid, WFListToPairOfPics(WFSetList, N))

def generateWFData(amount = 100, N=201):
    WFData = []
    for counter in range(amount):
        print(counter)
        # if counter > amount//2:
        #     randSize = np.random.randint(2, 4)
        #     shape = generatePolygon(randSize)
        #     WFSetList = polygonToWFsetList(shape, gridSize=N, angleAccuracy=360)
        # else:
        shape = genEll()
        WFSetList = ellipseToWFsetList(shape, gridSize=N, angleAccuracy=360)
        WF = dim3WFList(WFSetList)
        SinoWF = dim3getSinoWFFromList(WFSetList, N=N)
        arr = [np.array([SinoWF[j][0], SinoWF[j][1], WF[j][0], WF[j][1], SinoWF[j][2], WF[j][2]]) for j in range(len(WF))]
        WFData.extend(arr)
    return np.array(WFData)

seed = 52
np.random.seed(seed)

amount = 100

data = generateWFData(amount=amount)
# np.savetxt(f"nHeu{amount}_{seed}.txt", data, delimiter=' ', newline='\n', fmt='%d')
# edgs = []


# edgs2 = [f"{k:d} {j:d}" if np.linalg.norm((data[k]-data[j])) >= 2 else '' for j in range(len(data)) for k in range(j)]

# # for j in range(len(data)):
# #     for k in range(j):
# #         if np.linalg.norm((data[k]-data[j])/200) >= 1/100:
# #             edgs.append((k,j))
# #edgArr = np.array(edgs2)
# np.savetxt(f"Full{amount}.txt", edgs2, newline='\n', fmt='%s')

# tic = time.perf_counter()
# feffer.submanifoldInterpolation(4, 200, data)
# toc = time.perf_counter()
# print(f"Submanifold find took {toc - tic:0.4f} seconds\n")

