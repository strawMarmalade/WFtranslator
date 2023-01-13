import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
import time

import torch



def generateEllipse(smallestSize=10e-5, midRad=0.1, offCenter=True):
    maxA = 1.0; maxB = 1.0
    x0 = 0; y0 = 0
    if offCenter:
        (x0,y0) =  np.random.uniform(-1+midRad+smallestSize, 1-smallestSize-midRad, 2)
        maxA = np.min([1-x0,1+x0])
        maxB = np.min([1-y0,1+y0])
    a = np.random.uniform(np.min([midRad,smallestSize/2]), maxA-smallestSize)
    b = np.random.uniform(np.min([midRad,smallestSize/2]), maxB-smallestSize)
    return (a,b,x0,y0)

def drawEllipse(ell,stepSize=1000,output=True):
    a,b,x0,y0 = ell
    x = np.linspace(-a, a, stepSize)
    yPlus = [b*np.sqrt(1-(val/a)**2)+y0 for val in x]
    yMinus = [-b*np.sqrt(1-(val/a)**2)+y0 for val in x]
    xnew = [x0+val for val in x]
    plt.plot(xnew,yPlus)
    plt.plot(xnew,yMinus)
    if output:
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.show()

def generatePolygon(pointNum, smallestSize=10e-5, niceness=0.1, minRad=0.1, offCenter=True):
    """

    Parameters
    ----------
    pointNum : float
        desired number of vertices -1
    smallestSize : float, optional
        a machine epsilon type thing to stay away from boundaries, optional
        The default is 10e-5.
    niceness : float, optional
        has to do with the spikyness of the polygon, between 0 and pi, doesnt actually prevent spikyness 100% either
    minRad : float, optional
        try to stay this far away from boundary and try to let individual edges be at least this large
    offCenter : bool, optional
        if generate polygon with non-start-point (0,0). The default is True.

    Returns
    -------
    points : array of float pairs
        pointNum +2 vertices that in this order define the polygon
        start and end point are the same
    """
    phis = []
    phis.append(np.random.uniform(0, np.pi-niceness))
    for val in range(pointNum):
        new = phis[val]+np.random.uniform(niceness/2, np.pi-niceness)
        if new>2*np.pi:
            new -= 2*np.pi
        phis.append(new)
    phis = np.sort(phis)
    if offCenter:
        (x0,y0) =  np.random.uniform(-1+minRad, 1-minRad, 2)
        points = []
        for val in range(pointNum):
            maxRad = 1.0
            phi = phis[val]
            if (3/2)*np.pi<phi or phi<(1/2)*np.pi:
                maxRad = np.min([1,(1-x0)*(np.cos(phi))**(-1)])
            if (3/2)*np.pi>phi>(1/2)*np.pi:
                maxRad = np.min([1,-(1+x0)*(np.cos(phi))**(-1)])
            if 0<phi<np.pi:
                maxRad = np.min([maxRad, (1-y0)*(np.sin(phi))**(-1)])
            if np.pi<phi<2*np.pi:
                maxRad = np.min([maxRad, -(1+y0)*(np.sin(phi))**(-1)])
            rad = np.random.uniform(np.min([(maxRad-smallestSize)/2,minRad+smallestSize]),maxRad-smallestSize)
            points.append((x0+rad*(np.cos(phis[val])),y0+rad*(np.sin(phis[val]))))
        points.insert(0,(x0,y0))
        points.append((x0,y0))
    else:
        rads = np.random.uniform(smallestSize+minRad, 1-smallestSize, pointNum)
        points = [(rads[val]*np.cos(phis[val]),rads[val]*np.sin(phis[val])) for val in range(pointNum)]
        points.insert(0,(0,0))
        points.append((x0,y0))
    return np.array(points)

def drawPolygon(points, output=True):
    for val in range(len(points)-1):
        plt.plot([points[val][0],points[val+1][0]],[points[val][1],points[val+1][1]], color='blue')
    if output:
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.show()

def pointToGridIndex(x,y,gridSize):
    return [np.round((gridSize-1)/2*(x+1)).astype(int),np.round((gridSize-1)/2*(y+1)).astype(int)]

def gridIndexToPoint(x,y,gridSize):
    return np.array([2/(gridSize-1)*x-1,2/(gridSize-1)*y-1])

def ellipseToWFsetList(ell,gridSize=200, angleAccuracy=360):
    a, b, x0, y0 = ell
    x = np.linspace(-a, a, gridSize*2,endpoint=True)
    yPlus = [b*np.sqrt(1-a**(-2)*(val**2))+y0 for val in x]
    yMinus = [-b*np.sqrt(1-a**(-2)*(val**2))+y0 for val in x]
    #we allow the below things to divide by zero because
    #arctan2 can handle when one of the parameters is infinity
    #but I dont want to have to see the warnings so I supress them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anglesPlus = [np.round((np.arctan2(1,2*b*(a**(-2))*val/(np.sqrt(1-(val/a)**2))))*angleAccuracy/(2*np.pi)).astype(int)%angleAccuracy for val in x]
        anglesMinus = [np.round((np.arctan2(-1,2*b*(a**(-2))*val/(np.sqrt(1-(val/a)**2))))*angleAccuracy/(2*np.pi)).astype(int)%angleAccuracy for val in x]
    xnew = [x0+val for val in x]

    WFSetList = [[pointToGridIndex(xnew[val],yPlus[val],gridSize),[anglesPlus[val]]] for val in range(len(x))]
    WFSetList.extend([[pointToGridIndex(xnew[val],yMinus[val],gridSize),[anglesMinus[val]]] for val in range(len(x))])
    
    return WFSetList

def polygonToWFsetList(poly, gridSize=200, angleAccuracy=360):
    WFSetList = []
    #as the last element of the polygon is the first element, we look at the second last element
    #and take the edge from there to the first vertex
    towardPointLineMid = 0.5*poly[0]-0.5*poly[-2]
    for val in range(len(poly)-1):
        awayPointLineMid = 0.5*poly[val+1]-0.5*poly[val]
        #arctan2 gives angle between -pi and pi 
        towardAngle = np.arctan2(towardPointLineMid[1],towardPointLineMid[0])
        #outwardNormalTowardAngle = towardAngle-np.pi/2
        #inwardNormalTowardAngle = towardAngle +np.pi/2
        awayAngle = np.arctan2(awayPointLineMid[1],awayPointLineMid[0])
        outwardNormalAwayAngle = awayAngle-np.pi/2
        #inwardNormalAwayAngle = awayAngle+np.pi/2
        
        #outwardWFStartAngle = np.round((outwardNormalTowardAngle)*angleAccuracy/(2*np.pi)).astype(int)%angleAccuracy
        outwardWFEndAngle = np.round((outwardNormalAwayAngle)*angleAccuracy/(2*np.pi)).astype(int)%angleAccuracy
        
        # if outwardWFStartAngle< 0:
        #     outwardWFStartAngle = outwardWFStartAngle+angleAccuracy
        # if outwardWFEndAngle<0:
        #     outwardWFEndAngle= outwardWFEndAngle+angleAccuracy
        #this adds the wavefront outward angle to each intermediate point between
        #the two vertices
        distBetweenPoints = np.linalg.norm(poly[val]-poly[val+1])
        interPoints = np.round((gridSize-1)/2*distBetweenPoints)
        interPoints = interPoints.astype(int)
        for k in range(interPoints+1):
            stepTaken = poly[val]+k/interPoints*(poly[val+1]-poly[val])
            stepTakenAsGrid = pointToGridIndex(stepTaken[0], stepTaken[1], gridSize)
           # WFSetGrid[stepTakenAsGrid[0]][stepTakenAsGrid[1]][outwardWFEndAngle] = 1
            WFSetList.append([stepTakenAsGrid,[outwardWFEndAngle]])
        #the following list will be filled with every angle between the two outward pointing
        #angles from above which is the set of outward wavefront directions for a corner point
        #of a polygon     
        
        pointAsGrid = pointToGridIndex(poly[val][0],poly[val][1],gridSize)
        
        towardAngleBackward = (int(angleAccuracy/2)+np.round((towardAngle)*angleAccuracy/(2*np.pi)).astype(int))%angleAccuracy
        awayAngleDegrees = np.round((awayAngle)*angleAccuracy/(2*np.pi)).astype(int)%angleAccuracy
        if towardAngleBackward <= awayAngleDegrees:
            WFSetList.append([pointAsGrid,list(range(towardAngleBackward,awayAngleDegrees+1))])
        else:
            vals1 = list(range(towardAngleBackward,angleAccuracy))
            vals1.extend(list(range(awayAngleDegrees+1)))
            WFSetList.append([pointAsGrid,vals1])

        #when going to next point the one away line will turn into the toward line for the next point
        towardPointLineMid = awayPointLineMid
    return WFSetList
             
def drawWFSetList(WFSetList,gridSize=200, saveFile=True):
    for val in range(len(WFSetList)):
        pointGrid = WFSetList[val][0]
        point = gridIndexToPoint(pointGrid[0], pointGrid[1], gridSize)
        angles = WFSetList[val][1]
        for angle in angles:
            #to plot the WFset we just make small lines in the correct direction
            #at the point
            vec = [0.05*np.cos((2*np.pi*angle/360)), 0.05*np.sin((2*np.pi*angle/360))]
            plt.plot([point[0],point[0]+vec[0]],[point[1],point[1]+vec[1]],color='black',linewidth=0.3)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    if saveFile:
        plt.savefig('file.png',dpi=300)
    plt.show()

def convertWFListToWFGridLeoConvetion(List, gridSize=200, angleAccuracy=360):
    LeoAngleAcc = int(angleAccuracy/2)
    WFSetGrid = np.zeros([gridSize,gridSize, int(LeoAngleAcc)])
    for val in List:
        point = val[0]
        angleListHalf = [ang%LeoAngleAcc for ang in val[1]]
        WFSetGrid[point[0],point[1], angleListHalf] = 1
    return WFSetGrid

def checkIfPointInTriangle(p, xs):
    #give xs as 4 points where xs[0]=xs[3] 
    #so that its easier to do
    for val in range(3):
        #line from x0 to x1 is (1-t)x0+tx1
        #shifting the origin we get the vector v=(x1-x0)
        v = xs[val+1]-xs[val]
        #ortho. proj. of point p onto (1-t)x0+tx1 is
        orthP = np.outer(v, v)@(p-xs[val+1])/np.dot(v,v) + xs[val+1]
        #outer normal vector along that line is
        vNormal = np.arctan2(v[1],v[0]) - np.pi/2
        #distance between original point p and its orth projection is
        dist = np.linalg.norm(p-orthP)
        #this vector points away from p if p is inside the triangle and 
        #points toward p if p is outside of the triangle
        normalAsVec = [dist/2*np.cos(vNormal), dist/2*np.sin(vNormal)]
        #so start at orthP, walk the outer normal, and check if you've gotten closer to 
        #p than you were before. If so, p is outside of the triangle
        if np.linalg.norm((orthP+normalAsVec)-p) < dist:
            return False
    return True
    
def checkIfPointIsInPolygon(p, poly):
    #as the first and last element in the list poly is the same starting point, we leave it away
    for index in range(1,len(poly)-2):
        #the only polgons we can construct are star-convex so we just have to check for
        #all the triangles that make up the polygon
        if checkIfPointInTriangle(p, [poly[0],poly[index],poly[index+1],poly[0]]):
            return True
    return False

def gridOfAllPointsInTriangle(xs, gridSize=200, angleAccuracy=360):
    grid = np.zeros([gridSize,gridSize],dtype=bool)
    #take again x3=x0
    #lets find at least one other angle less than 90 degrees
    towardPointLineMid = 0.5*(xs[0]-xs[-2])
    awayPointLineMid = 0.5*(xs[1]-xs[0])
    #arctan2 gives angle between -pi and pi 
    towardAngle = np.arctan2(towardPointLineMid[1],towardPointLineMid[0])
    towardAngleDegrees = np.round(towardAngle*angleAccuracy/(2*np.pi)).astype(int)%angleAccuracy

    awayAngle = np.arctan2(awayPointLineMid[1],awayPointLineMid[0])
    awayAngleDegrees = np.round((awayAngle)*angleAccuracy/(2*np.pi)).astype(int)%angleAccuracy

    pointFrom = xs[0]
    pointTo = xs[1]
    if (towardAngleDegrees+int(angleAccuracy/2)-awayAngleDegrees)%angleAccuracy > int(angleAccuracy/4):
        #the angle at the origin is greater than 90 so 
        #we will use the other points for the walk
        pointFrom = xs[1]
        pointTo = xs[2]
    else:
        #we have to find the other angle with less than 90 degrees
        towardPointLineMid = 0.5*(xs[1]-xs[0])
        awayPointLineMid = 0.5*(xs[2]-xs[1])
        #arctan2 gives angle between -pi and pi 
        #towardAngle = np.arctan2(towardPointLineMid[1],towardPointLineMid[0])
        #towardAngleDegrees = np.round(towardAngle*angleAccuracy/(2*np.pi)).astype(int)%angleAccuracy
        
        #weve moved to the next vertex so no need to calc again
        towardAngleDegrees = awayAngleDegrees
        
        awayAngle = np.arctan2(awayPointLineMid[1],awayPointLineMid[0])
        awayAngleDegrees = np.round((awayAngle)*angleAccuracy/(2*np.pi)).astype(int)%angleAccuracy

        #if this is true then the angle from x1 to x2 is greater than 90 degrees
        if (towardAngleDegrees+int(angleAccuracy/2)-awayAngleDegrees)%angleAccuracy > int(angleAccuracy/4):
            #the line from x0 to x1 is unsuitable to build the grid so we use 
            #the line from x2 to x0
            pointFrom = xs[2]
            pointTo = xs[0]
            
    awayPointLineMid = 0.5*(pointTo-pointFrom)

    awayAngle = np.arctan2(awayPointLineMid[1],awayPointLineMid[0])
    inwardNormalAwayAngle = awayAngle+np.pi/2
    #inwardNormalAngle = (-np.round((outwardNormalAwayAngle)*angleAccuracy/(2*np.pi)).astype(int))%angleAccuracy

    #we will now walk along the path from pointFrom to pointTo
    #and walk in the direction of the inward normal and add points
    #to the grid as long as we dont land outside of the triangle
    distBetweenPoints = np.linalg.norm(pointFrom-pointTo)
    interPoints = np.round((gridSize-1)*distBetweenPoints)
    interPoints = interPoints.astype(int)
    for k in range(interPoints+1):
        #here we are walking along the line pointFrom to pointTo
        stepTaken = pointFrom+k/interPoints*(pointTo-pointFrom)
        stepTakenAsGrid = pointToGridIndex(stepTaken[0], stepTaken[1], gridSize)
        grid[stepTakenAsGrid[1],stepTakenAsGrid[0]] = True
        
        #here we additionally walk along the inward normal and add points to the grid
        stepInNormal = np.array([1/(2*gridSize)*np.cos(inwardNormalAwayAngle), 1/(2*gridSize)*np.sin(inwardNormalAwayAngle)])
        stepAlongNormalHere = stepTaken+stepInNormal
        while checkIfPointInTriangle(stepAlongNormalHere, xs):
            gridIndex = pointToGridIndex(stepAlongNormalHere[0], stepAlongNormalHere[1], gridSize)
            grid[gridIndex[1],gridIndex[0]] = True
            stepAlongNormalHere = stepAlongNormalHere+stepInNormal
            
    return grid


def gridOfAllPointsInPolygon(poly, gridSize=200, angleAccuracy=360):
    grid = np.zeros([gridSize,gridSize],dtype=int)
    #as the first and last element in the list poly is the same starting point, we leave it away
    for index in range(1,len(poly)-2):
        #the only polgons we can construct are star-convex so we just have to check for
        #all the triangles that make up the polygon
        grid = np.logical_or(grid, gridOfAllPointsInTriangle([poly[0],poly[index],poly[index+1],poly[0]],gridSize=gridSize,angleAccuracy=angleAccuracy))
        #if checkIfPointInTriangle(p, [poly[0],poly[index],poly[index+1],poly[0]]):
    return grid > 0

def checkIfPointInEllipse(p, ell):
    a, b, x0, y0 = ell
    if -a <= p[0]-x0 <= a:
        if -b*np.sqrt(1-((p[0]-x0)/a)**2)+y0 <= p[1] <= b*np.sqrt(1-((p[0]-x0)/a)**2)+y0:
            return True
    return False

def constructImageInGridOfEllipse(ell, gridSize=200):
    return np.array([[checkIfPointInEllipse(gridIndexToPoint(j,k, gridSize),ell) for j in range(gridSize)] for k in range(gridSize)], dtype=bool)

def constructImageInListOfEllipse(ell, gridSize=200):
    listOfPoints = []
    for k in range(gridSize):
        for j in range(gridSize):
            #this is point in the 'real' coordinates
            point = gridIndexToPoint(k,j, gridSize)
            if checkIfPointInEllipse(point,ell):
                listOfPoints.append(point)
    return listOfPoints

def constructImageInGridOfPolygon(poly, gridSize=200):
    return np.array([[checkIfPointIsInPolygon(gridIndexToPoint(j,k, gridSize),poly) for j in range(gridSize)] for k in range(gridSize)], dtype=bool)

def drawGrid(grid):
    plt.imshow(grid, origin='lower') 
    plt.show()


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
    drawPolygon(poly, output=True)
    drawWFSetList(WFSetList, gridSize=gridSize, saveFile=False)
    toc = time.perf_counter()
    print(f"Plotting polygon with wavefrontset picture took {toc - tic:0.4f} seconds")
    
    #This is the old method that is one order of magnitude smaller than the 
    #new method, used below
    # tic = time.perf_counter()
    # grid1 = constructImageInGridOfPolygon(poly, gridSize=gridSize)
    # toc = time.perf_counter()
    # print(f"Get inside of polygon as grid old way took {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    grid2 = gridOfAllPointsInPolygon(poly, gridSize=gridSize, angleAccuracy=angleAccuracy)
    toc = time.perf_counter()
    print(f"Get inside of polygon way as grid took {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    #drawGrid(grid1)
    drawGrid(grid2)
    #drawGrid(np.logical_xor(grid1,grid2))
    toc = time.perf_counter()
    print(f"Drawing the grid of polygon took {toc - tic:0.4f} seconds\n")
    

def fullEllipseRoutineTimer(gridSize = 200):   
    print(f"Grid size is {gridSize:d}")

    tic = time.perf_counter()
    ell = generateEllipse()
    toc = time.perf_counter()
    print(f"Ellipse generation took {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    WFSetList = ellipseToWFsetList(ell)
    toc = time.perf_counter()
    print(f"Wavefrontset calculation took {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    drawEllipse(ell, output=False)
    drawWFSetList(WFSetList, gridSize=gridSize, saveFile=False)
    toc = time.perf_counter()
    print(f"Plotting ellipse with wavefrontset picture took {toc - tic:0.4f} seconds")
    
    tic = time.perf_counter()
    grid = constructImageInGridOfEllipse(ell, gridSize=gridSize)
    toc = time.perf_counter()
    print(f"Get inside of ellipse as grid took {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    drawGrid(grid)
    toc = time.perf_counter()
    print(f"Drawing the grid of polygon took {toc - tic:0.4f} seconds\n")

fullPolygonRoutineTimer(polySize=5)
fullEllipseRoutineTimer()


#ell = generateEllipse()
#drawEllipse(ell, output=True)
#drawPolygon(poly, output=False)
# grid = constructImageInGridOfPolygon(poly)
# #print(List)
# drawGrid(grid)

#print(checkIfPointInEllipse(p, ell))


### Leo stuff
# N = 201

# ImageWF = convertWFListToWFGridLeoConvetion(List, gridSize=201)
# #ImageWF = np.zeros([N, N, 180])

# #this block sets some test images for sanity check
# # j=0
# # while j <= N-1:
# #     ImageWF[j, 1, 0] = 1
# #     j = j+1


# #Actual code for WF mapping starts here

# #initializing sinogram WF
# SinoWF = np.zeros([N,180,180])
def canonicalplus1(r, alpha, phi):
    return (math.acos(r*math.cos(alpha - phi)) + phi)
def canonicalminus1(r, alpha, phi):
    return (-math.acos(r*math.cos(alpha - phi)) + phi)
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
    #print(JMatrix)
    #print(SinoWFVect)
#    print(math.degrees(math.acos(SinoWFVect[0]/math.sqrt(SinoWFVect[0]**2 + SinoWFVect[1]**2))))
    return round(math.degrees(math.acos(SinoWFVect[0]/math.sqrt(SinoWFVect[0]**2 + SinoWFVect[1]**2))))




# rowindex = 0
# while (rowindex <= N-1):
#     colindex = 0
#     while (colindex <= N-1):
#         WFangleindex = 0
#         while (WFangleindex <= 179):
#             if ImageWF[rowindex, colindex, WFangleindex] ==1:
#                radius = math.sqrt((2*rowindex/(N-1) -1)**2 + (2*colindex/(N-1) - 1)**2)
#                #computes the distance of the pixel from the origin
#                if radius ==0:
#                    positionangle = 0
#                else:
#                    #print((2 * colindex / (N - 1) - 1) / radius)
#                    #print(math.acos((2 * colindex / (N - 1) - 1) / radius))
#                    positionangle = np.sign((2*rowindex/(N-1) -1))*math.acos((2 * colindex / (N - 1) - 1) / radius)
#                    if np.sign((2*rowindex/(N-1) -1))== 0:
#                        positionangle = math.acos((2 * colindex / (N - 1) - 1) / radius)
#                    #print(positionangle)
#                #positionangle is the angle of the position measured in Radians. It takes the range between -pi to pi
#                WFangleradian = math.radians(WFangleindex)
#                #turns WFangle from entry index to radians
#                boundaryradplus = canonicalplus1(radius, positionangle, WFangleradian)
#                #above function returns location on the boundary of circle in radians.
#                # So the range is a float between 0 and 2pi
#                boundarydegreeplus = math.degrees(boundaryradplus)
#                boundaryindexplus = round(boundarydegreeplus *N/360)%N
#                boundaryradminus = canonicalminus1(radius, positionangle, WFangleradian)
#                boundarydegreeminus = math.degrees(boundaryradminus)
#                boundaryindexminus = round(boundarydegreeminus *N/360)%N
#                incomingradplus = canonicalplus2(radius, positionangle, WFangleradian)
#                #above function returns incoming direction in radians relative to the inward pointing normal
#                #so the range is an integer between -pi/2 degrees to pi/2 degrees
#                incomingdegreeplus = math.degrees(incomingradplus)
#                incomingradminus = - incomingradplus
#                incomingdegreeminus = -incomingdegreeplus
#                incomingindexplus = round(incomingdegreeplus + 90)%180
#                incomingindexminus = round(incomingdegreeminus + 90)%180
#                tplus = traveltimeplus(radius, positionangle, WFangleradian)
#                tminus = traveltimeminus(radius, positionangle, WFangleradian)
#                SinoWFindexplus = pullback(boundaryradplus, incomingradplus, WFangleradian, tplus)
#                SinoWFindexminus = pullback(boundaryradminus, incomingradminus, WFangleradian, tminus)
#                SinoWF[boundaryindexplus, incomingindexplus, SinoWFindexplus] = 1
#                SinoWF[boundaryindexminus, incomingindexminus, SinoWFindexminus] = 1
#             WFangleindex = WFangleindex + 1
#         colindex = colindex + 1
#     rowindex = rowindex + 1        

# print(SinoWF)
# SinoWFtensor = torch.tensor(SinoWF)
# print(torch.nonzero(SinoWFtensor))
    