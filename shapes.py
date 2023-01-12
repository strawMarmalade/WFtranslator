import numpy as np
import matplotlib.pyplot as plt

def generateEllipse(smallestSize=10e-5, offCenter=True):
    maxA = 1.0; maxB = 1.0
    x0 = 0; y0 = 0
    if offCenter:
        (x0,y0) =  np.random.uniform(-1+smallestSize, 1-smallestSize, 2)
        maxA = np.min([1-x0,1+x0])
        maxB = np.min([1-y0,1+y0])
    a = np.random.uniform(smallestSize/2, maxA-smallestSize)
    b = np.random.uniform(smallestSize/2, maxB-smallestSize)
    return (a,b,x0,y0)

def drawEllipse(a,b,x0,y0,stepSize=1000):
    x = np.linspace(-a, a, stepSize)
    yPlus = [b*np.sqrt(1-a**(-2)*(val**2))+y0 for val in x]
    yMinus = [-b*np.sqrt(1-a**(-2)*(val**2))+y0 for val in x]
    xnew = [x0+val for val in x]
    plt.plot(xnew,yPlus)
    plt.plot(xnew,yMinus)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.show()
    
#ell = generateEllipse()
#drawEllipse(ell[0], ell[1], ell[2], ell[3], 200)

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
    #plt.plot([points[-1][0],points[0][0]],[points[-1][1],points[0][1]], color='red')
    #plt.scatter(points[0][0],points[0][1])
    #plt.scatter(points[-1][0],points[-1][1], color='green')
    if output:
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.show()

def pointToGridIndex(x,y,gridSize):
    return [np.round((gridSize-1)/2*(x+1)).astype(int),np.round((gridSize-1)/2*(y+1)).astype(int)]

def gridIndexToPoint(x,y,gridSize):
    return [2/(gridSize-1)*x-1,2/(gridSize-1)*y-1]

def polygonToWFsetGrid(poly, gridSize=200, angleAccuracy=360):
    #WFSetGrid = np.zeros(gridSize,gridSize,angleAccuracy)
    WFSetList = []
    #as the last element of the polygon is the first element, we look at the second last element
    #and take the edge from there to the first vertex
    towardPointLineMid = 0.5*poly[0]-0.5*poly[-2]
    for val in range(len(poly)-1):
        awayPointLineMid = 0.5*poly[val+1]-0.5*poly[val]
        #arctan2 gives angle between -pi and pi 
        towardAngle = np.arctan2(towardPointLineMid[1],towardPointLineMid[0])
        outwardNormalTowardAngle = towardAngle-np.pi/2
        #inwardNormalTowardAngle = towardAngle - np.pi/2
        awayAngle = np.arctan2(awayPointLineMid[1],awayPointLineMid[0])
        outwardNormalAwayAngle = awayAngle-np.pi/2
        #inwardNormalAwayAngle = awayAngle-np.pi/2
        
        outwardWFStartAngle = np.round((outwardNormalTowardAngle)*angleAccuracy/(2*np.pi)).astype(int)
        outwardWFEndAngle = np.round((outwardNormalAwayAngle)*angleAccuracy/(2*np.pi)).astype(int)
        
        if outwardWFStartAngle< 0:
            outwardWFStartAngle = outwardWFStartAngle+angleAccuracy
        if outwardWFEndAngle<0:
            outwardWFEndAngle= outwardWFEndAngle+angleAccuracy
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
        #for val in range(outwardWFStartAngle,outwardWFEndAngle+1):
        #    WFSetGrid[pointAsGrid[0]][pointAsGrid[1]][val] = 1
        
        #if outwardWFEndAngle >= outwardWFStartAngle:
        #    WFSetList.append([pointAsGrid,[val2 for val2 in range(outwardWFStartAngle,outwardWFEndAngle+1)]])
        towardAngleBackward = (int(angleAccuracy/2)+np.round((towardAngle)*angleAccuracy/(2*np.pi)).astype(int))%angleAccuracy
        awayAngleDegrees = np.round((awayAngle)*angleAccuracy/(2*np.pi)).astype(int)%angleAccuracy
        if towardAngleBackward <= awayAngleDegrees:
            WFSetList.append([pointAsGrid,[val2 for val2 in range(towardAngleBackward,awayAngleDegrees+1)]])
        else:
            vals1 = [val2 for val2 in range(towardAngleBackward,angleAccuracy+1)]
            vals1.extend([val2 for val2 in range(awayAngleDegrees+1)])
            #WFSetList.append([pointAsGrid,[val2 for val2 in range(awayAngleDegrees,towardAngleBackward+1)]])
            WFSetList.append([pointAsGrid,vals1])
            # vals1 = [val for val in range(outwardWFStartAngle,angleAccuracy+1)]
            # vals1.extend([val for val in range(outwardWFEndAngle)])
            # WFSetList.append([pointAsGrid,vals1])
            #WFSetList.append([pointAsGrid,[val for val in range(outwardWFEndAngle,outwardWFStartAngle+1)]])
        #when going to next point the one away line will turn into the toward line for the next point
        towardPointLineMid = awayPointLineMid
    return WFSetList
             


def drawWFSetList(WFSetList,gridSize=200):
    for val in range(len(WFSetList)):
        pointGrid = WFSetList[val][0]
        point = gridIndexToPoint(pointGrid[0], pointGrid[1], gridSize)
        angles = WFSetList[val][1]
        #plt.scatter(point[0],point[1], color='blue')
        for angle in angles:
            vec = [0.05*np.cos((2*np.pi*angle/360)), 0.05*np.sin((2*np.pi*angle/360))]
            plt.plot([point[0],point[0]+vec[0]],[point[1],point[1]+vec[1]],color='black',linewidth=0.3)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.savefig('images/file.png',dpi=300)
    plt.show()

poly = generatePolygon(5)
drawPolygon(poly, output=False)
List = polygonToWFsetGrid(poly)
drawWFSetList(List)
    
         
         
    