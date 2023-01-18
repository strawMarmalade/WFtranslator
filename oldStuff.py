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


def gridOfAllPointsInPolygon(poly, gridSize=200, angleAccuracy=360):
    grid = np.zeros([gridSize,gridSize],dtype=int)
    #as the first and last element in the list poly is the same starting point, we leave it away
    for index in range(1,len(poly)-2):
        #the only polgons we can construct are star-convex so we just have to check for
        #all the triangles that make up the polygon
        grid = np.logical_or(grid, gridOfAllPointsInTriangle([poly[0],poly[index],poly[index+1],poly[0]],gridSize=gridSize,angleAccuracy=angleAccuracy))
        #if checkIfPointInTriangle(p, [poly[0],poly[index],poly[index+1],poly[0]]):
    return grid > 0

def toAngAcc(angle,angleAccuracy=360):
    return int(angle/2/np.pi * angleAccuracy)%angleAccuracy

def fromAngAcc(deg,angleAccuracy=360):
    return deg/angleAccuracy*2*np.pi

def radonTrafo(grid,gridSize=200, angleAccuracy=360):
    radonValue = np.zeros([gridSize,gridSize,int(angleAccuracy/2)])
    for angleIndexAdd in range(angleAccuracy):
        print(angleIndexAdd)
        angle = 2*np.pi*(angleAccuracy+angleIndexAdd)
        pointOnOuterCircle = np.array([2*np.cos(angle),2*np.sin(angle)])
        #now the inward angle is angle+pi
        innerNormal = angle+np.pi
        normalDeg = toAngAcc(innerNormal,angleAccuracy)
        for angleOffset in range(-int(angleAccuracy/4),int(angleAccuracy/4)):
            walkAngleDeg = normalDeg+angleOffset
            walkAngle = fromAngAcc(walkAngleDeg,angleAccuracy)
            steps = int(gridSize*3)
            walkVector = np.array([2/steps*np.cos(walkAngle),2/steps*np.sin(walkAngle)])
            #currentSpot = pointOnOuterCircle
            currentSpotAsGrid = pointToGridIndex(pointOnOuterCircle[0], pointOnOuterCircle[1], gridSize)
            for k in range(steps+1):
                newSpot = gridIndexToPoint(currentSpotAsGrid[0], currentSpotAsGrid[1], gridSize) + walkVector
                newSpotAsGrid = pointToGridIndex(newSpot[0],newSpot[1], gridSize)
                if newSpotAsGrid[0] <= 199 and newSpotAsGrid[1] <= 199 and grid[newSpotAsGrid[0],newSpotAsGrid[1]] and newSpotAsGrid != currentSpotAsGrid:
                    radonValue[newSpotAsGrid[0],newSpotAsGrid[1],walkAngleDeg] += 1
                currentSpotAsGrid = newSpotAsGrid
    return radonValue

#below is the better one
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
        while newCheckIfInTriangle(stepAlongNormalHere, np.array(xs)):
            gridIndex = pointToGridIndex(stepAlongNormalHere[0], stepAlongNormalHere[1], gridSize)
            grid[gridIndex[1],gridIndex[0]] = True
            stepAlongNormalHere = stepAlongNormalHere+stepInNormal
            
    return grid

def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end
    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        if x != x1 and x < x2-2:
            if ystep == 1:
                y2 = y-2
            else:
                y2 = y+2
    
            coord2 = (y2, x) if is_steep else (x, y2)
            points.append(coord2)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points



def gridOfAllPointsInTriangle2(xs, gridSize=200, angleAccuracy=360):
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
    interPoints = np.round(2*(gridSize-1)*distBetweenPoints)
    interPoints = interPoints.astype(int)
    for k in range(interPoints+1):
        #here we are walking along the line pointFrom to pointTo
        stepTaken = pointFrom+k/interPoints*(pointTo-pointFrom)
        stepTakenAsGrid = pointToGridIndex(stepTaken[0], stepTaken[1], gridSize)
        grid[stepTakenAsGrid[1],stepTakenAsGrid[0]] = True
        
        #here we additionally walk along the inward normal and add points to the grid
        stepInNormal = np.array([1/(2*gridSize)*np.cos(inwardNormalAwayAngle), 1/(2*gridSize)*np.sin(inwardNormalAwayAngle)])
        stepAlongNormalHere = stepTaken+stepInNormal
        while newCheckIfInTriangle(stepAlongNormalHere, np.array(xs)):
            stepAlongNormalHere = stepAlongNormalHere+stepInNormal
        stepEndAsGrid = pointToGridIndex(stepAlongNormalHere[0], stepAlongNormalHere[1], gridSize)
        allPointsOnLine = get_line(stepTakenAsGrid, stepEndAsGrid)
        for val in allPointsOnLine:
            grid[val[1],val[0]] = True
            
    return grid


def polyGrid1(poly, gridSize=200):
    grid = np.zeros([gridSize,gridSize],dtype=bool)
    for j in range(gridSize):
        for k in range(gridSize):
            p = gridIndexToPoint(j, k, gridSize)
            if np.min(poly[:,0]) <= p[0] <= np.max(poly[:,0]) and np.min(poly[:,1]) <= p[1] <= np.max(poly[:,1]):
                grid[k,j] = checkIfPointIsInPolygonNew1(p,poly)
    return grid

def polyGrid2(poly, gridSize=200):
    grid = np.zeros([gridSize,gridSize],dtype=bool)
    for j in range(gridSize):
        for k in range(gridSize):
            p = gridIndexToPoint(j, k, gridSize)
            if np.min(poly[:,0]) <= p[0] <= np.max(poly[:,0]) and np.min(poly[:,1]) <= p[1] <= np.max(poly[:,1]):
                grid[k,j] = checkIfPointIsInPolygonNew2(p,poly)
    return grid
                
def polyGrid3(poly, gridSize=200):
    grid = np.zeros([gridSize,gridSize],dtype=bool)
    for j in range(gridSize):
        for k in range(gridSize):
            p = gridIndexToPoint(j, k, gridSize)
            if np.min(poly[:,0]) <= p[0] <= np.max(poly[:,0]) and np.min(poly[:,1]) <= p[1] <= np.max(poly[:,1]):
                grid[k,j] = checkIfPointIsInPolygonNew3(p,poly)
    return grid


def checkIfPointIsInPolygonNew3(p, poly):
    #as the first and last element in the list poly is the same starting point, we leave it away
    for index in range(1,len(poly)-2):
        xs = np.array([poly[0],poly[index],poly[index+1],poly[0]])
        if np.min(xs[:,0]) <= p[0] <= np.max(xs[:,0]) and np.min(xs[:,1]) <= p[1] <= np.max(xs[:,1]):
        #the only polgons we can construct are star-convex so we just have to check for
        #all the triangles that make up the polygon
            if check2(p, xs):
                return True
    return False


def checkIfPointIsInPolygonNew1(p, poly):
    #as the first and last element in the list poly is the same starting point, we leave it away
    for index in range(1,len(poly)-2):
        xs = np.array([poly[0],poly[index],poly[index+1],poly[0]])
        if np.min(xs[:,0]) <= p[0] <= np.max(xs[:,0]) and np.min(xs[:,1]) <= p[1] <= np.max(xs[:,1]):
        #the only polgons we can construct are star-convex so we just have to check for
        #all the triangles that make up the polygon
            if checkIfPointInTriangle(p, xs):
                return True
    return False

def gridOfAllPointsInPolygon2(poly, gridSize=200, angleAccuracy=360):
    grid = np.zeros([gridSize,gridSize],dtype=int)
    #as the first and last element in the list poly is the same starting point, we leave it away
    for index in range(1,len(poly)-2):
        #the only polgons we can construct are star-convex so we just have to check for
        #all the triangles that make up the polygon
        grid = np.logical_or(grid, gridOfAllPointsInTriangle2([poly[0],poly[index],poly[index+1],poly[0]],gridSize=gridSize,angleAccuracy=angleAccuracy))
        #if checkIfPointInTriangle(p, [poly[0],poly[index],poly[index+1],poly[0]]):
    return grid > 0

def check(xs):
    return (xs[0,0] - xs[2,0]) * (xs[1,1] - xs[2,1]) - (xs[1,0] - xs[2,0]) * (xs[0,1] - xs[2,1])

def check2(p, xs):
    d1 = check(np.array([p,xs[0],xs[1]]))
    d2 = check(np.array([p,xs[1],xs[2]]))
    d3 = check(np.array([p,xs[2],xs[3]]))

    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    
    return not (has_neg and has_pos)

def constructImageInGridOfPolygon(poly, gridSize=200):
    return np.array([[checkIfPointIsInPolygon(gridIndexToPoint(j,k, gridSize),poly) for j in range(gridSize)] for k in range(gridSize)], dtype=bool)


def getmaxWH(center, angle):
    if center[0] > 0:
        xmax = 1-center[0]
    else:
        xmax = 1+center[0]
    if center[1] > 0:
        ymax = 1-center[1]
    else:
        ymax = 1+center[1]
    
    
    if angle == 0:
        return np.array([xmax,ymax])
    # else:
    #     bot = np.cos(2*angle)
    #     top1 = (ymax*np.sin(angle))**2
    #     top2 = (xmax*np.cos(angle))**2
    #     a = np.sqrt(((xmax*np.cos(angle))**2+ (ymax*np.sin(angle))**2)/np.abs(np.cos(2*angle)))
    #     b = np.sqrt(1/(xmax**2+ymax**2-a**(-2)))
    #     return np.array([1/a,b])
    
    
    # #this means the max width and height are identical, and so we'd divide by
    # #zero in the other formula so we need to check this first
    # if np.abs(angle - np.pi/4) < 10e-4 or np.abs(angle - 3*np.pi/4) < 10e-4:
    #     return np.array([xmax-10e-4, ymax - 10e-4])
    # else:
    #     a = np.sqrt(1/np.cos(2*angle))
    
    else:
        xmax2 = 1-center[0]
        ymax2 = -1-center[1]
        thing = (xmax**2-ymax**2)/np.cos(2*angle)
        a = np.sqrt(np.abs(xmax**2+ymax**2 + thing)/2)
        aa = np.sqrt(np.abs((xmax**2*np.cos(angle)**2 - ymax**2*np.sin(angle)**2)/(np.cos(2*angle))))
        bb = np.sqrt(np.abs(xmax**2+ymax**2+a**2))
        b =  np.sqrt(np.abs(xmax**2+ymax**2 - thing)/2)
        
        xmax2 = -1-center[0]
        ymax2 = -1-center[1]

        thing2 = (xmax2**2-ymax2**2)/np.cos(2*angle)
        a2 = np.sqrt(np.abs(xmax2**2+ymax2**2 + thing2)/2)
        aa2 = np.sqrt(np.abs((xmax2**2*np.cos(angle)**2 - ymax2**2*np.sin(angle)**2)/(np.cos(2*angle))))
        bb2 = np.sqrt(np.abs(xmax2**2+ymax2**2+a2**2))
        b2 =  np.sqrt(np.abs(xmax2**2+ymax2**2 - thing2)/2)
        
        #print(a,b,bb)
        return np.array([a,bb])
    
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

def genEll():
    angle = np.random.uniform(-1,1)*np.pi/4
    
    maxlen = np.sqrt(1+(np.tan(angle))**2)
    
    a = np.random.uniform(0,maxlen)
    
    maxb1 = np.sqrt((1-(a*np.cos(angle))**2)/(np.sin(angle)**2))
    maxb2 = np.sqrt((1-(a*np.sin(angle))**2)/(np.cos(angle)**2))
    
    maxb = np.min([maxb1,maxb2])

    b = np.random.uniform(0,maxb)
    
    #these are the maximal distances to border
    maxX = 1-np.sqrt((a*np.cos(angle))**2+(b*np.sin(angle))**2)
    maxY = 1-np.sqrt((a*np.sin(angle))**2+(b*np.cos(angle))**2)
    
    centerX = np.random.uniform(0,maxX)
    centerY = np.random.uniform(0,maxY)
    
    center = np.array([centerX,centerY])
    return (center, angle, a,b)



def gridEll(ell, gridSize=200):
    center, angle, a, b = ell
    
    maxX = np.sqrt((a*np.cos(angle))**2+(b*np.sin(angle))**2)
    maxY = np.sqrt((a*np.sin(angle))**2+(b*np.cos(angle))**2)
    
    botLeft = point2grid(np.array([center[0]-maxX,center[1]-maxY]))
    topRight = point2grid(np.array([center[0]+maxX,center[1]+maxY]))

    grid = np.zeros([gridSize,gridSize],dtype=bool)
    for j in range(botLeft[0],topRight[0]+1):
        for k in range(botLeft[1], topRight[1]+1):
            #print(j,k)
            #pp = grid2point(np.array([j,k]))
            grid[j,k] = checkIfPointInEllipse(grid2point(np.array([j,k])),ell)
    
    return grid

# def constructImageInGridOfEllipse(ell, gridSize=200):
#     return np.array([[checkIfPointInEllipse(gridIndexToPoint(j,k, gridSize),ell) for j in range(gridSize)] for k in range(gridSize)], dtype=bool)

# def constructImageInListOfEllipse(ell, gridSize=200):
#     listOfPoints = []
#     for k in range(gridSize):
#         for j in range(gridSize):
#             #this is point in the 'real' coordinates
#             point = gridIndexToPoint(k,j, gridSize)
#             if checkIfPointInEllipse(point,ell):
#                 listOfPoints.append(point)
#     return listOfPoints

def ellipseToWFsetList(ell,gridSize=200, angleAccuracy=360):
    a = ell.get_width()/2
    b = ell.get_height()/2
    x0,y0 = ell.get_center()
    angle = np.deg2rad(ell.get_angle())
    t = np.linspace(0, 2*np.pi, angleAccuracy)
    Ell = np.array([a*np.cos(t), b*np.sin(t)])  
    
    r = rot(angle)
    #2-D rotation matrix

    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = r@Ell[:,i]
    plt.xlim(-1,1)
    plt.ylim(-20,20)
    plt.plot(x0+Ell_rot[0,:] , y0+Ell_rot[1,:],'darkorange' )    #rotated ellipse
    # WFSetList = [[point2grid(np.array([x0+Ell_rot[0,j],y0+Ell_rot[1,j]])),[np.round((angle+np.arctan2(1,2*b*(a**(-2))*Ell[0,j]/(np.sqrt(1-(Ell[0,j]/a)**2))))*angleAccuracy/(2*np.pi)).astype(int)%angleAccuracy]] for j in range(angleAccuracy)]
    # x = np.linspace(-a, a, gridSize*2,endpoint=True)
    # yPlus = [b*np.sqrt(1-a**(-2)*(val**2))+y0 for val in x]
    # yMinus = [-b*np.sqrt(1-a**(-2)*(val**2))+y0 for val in x]
    #we allow the below things to divide by zero because
    #arctan2 can handle when one of the parameters is infinity
    #but I dont want to have to see the warnings so I supress them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anglesPlus = [j+np.rad2deg(angle) for j in range(angleAccuracy)]
        anglesOther = [np.rad2deg(angle+np.arctan(a*np.tan(j)/b)) for j in range(angleAccuracy)]
        #anglesPlus = [np.round((angle+np.arctan2(1,2*b*(a**(-2))*Ell[0,j]/(np.sqrt(1-(Ell[0,j]/a)**2))))*angleAccuracy/(2*np.pi)).astype(int)%angleAccuracy for j in range(angleAccuracy//2)]
        #anglesPlus.extend([np.round((angle+np.arctan2(-1,2*b*(a**(-2))*Ell[0,j]/(np.sqrt(1-(Ell[0,j]/a)**2))))*angleAccuracy/(2*np.pi)).astype(int)%angleAccuracy for j in range(angleAccuracy//2,angleAccuracy)])
    # xnew = [x0+val for val in x]

    WFSetList = [[point2grid(np.array([x0+Ell_rot[0,j],y0+Ell_rot[1,j]])),[anglesOther[j]]] for j in range(angleAccuracy)]
    #WFSetList.extend([[point2grid(np.array([x0+Ell_rot[0,j],y0+Ell_rot[1,j]])),[anglesMinus[val]]] for j in range(angleAccuracy//2,angleAccuracy)]
    
    return WFSetList

def genEllinBox():
    angle = np.random.uniform(-1,1)*np.pi/4
    
    maxlen = np.sqrt(1+(np.tan(angle))**2)
    a = np.random.uniform(0,maxlen)
    with warnings.catch_warnings():
        #I'm okay with dividing by zero here (which can only happen in maxb1)
        #because that just means I take it to be infinity
        warnings.simplefilter("ignore")
        maxb1 = np.sqrt((1-(a*np.cos(angle))**2)/(np.sin(angle)**2))
        maxb2 = np.sqrt((1-(a*np.sin(angle))**2)/(np.cos(angle)**2))
    
    maxb = np.min([maxb1,maxb2])

    b = np.random.uniform(0,maxb)
    
    #these are the maximal distances to border
    maxX = 1-np.sqrt((a*np.cos(angle))**2+(b*np.sin(angle))**2)
    maxY = 1-np.sqrt((a*np.sin(angle))**2+(b*np.cos(angle))**2)
            
    centerX = np.random.uniform(-maxX,maxX)
    centerY = np.random.uniform(-maxY,maxY)
    
    center = np.array([centerX,centerY])
    return Ellipse(center, 2*a,2*b, angle=np.rad2deg(angle))

def generatePolygonInBox(pointNum, smallestSize=10e-5, niceness=0.1, minRad=0.1, offCenter=True):
    phis = []
    phis.append(np.random.uniform(0, np.pi-niceness))
    for val in range(pointNum):
        new = phis[val]+np.random.uniform(niceness/2, np.pi-niceness)
        if new>2*np.pi:
            new -= 2*np.pi
        phis.append(new)
    phis = np.sort(phis)
    if offCenter:
        points = []
        rads = np.random.uniform(smallestSize, 1-smallestSize, pointNum)
        points2 = np.array([(rads[val]*np.cos(phis[val]),rads[val]*np.sin(phis[val])) for val in range(pointNum)])
        minX = np.min(points2[:,0])
        maxX = np.max(points2[:,0])
        minY = np.min(points2[:,1])
        maxY = np.max(points2[:,1])
        
        x0 = np.random.uniform(-np.min([1,1+minX]), np.min([1,1-maxX]))
        y0 = np.random.uniform(-np.min([1,1+minY]), np.min([1,1-maxY]))
        points2 += np.array([x0,y0])

        points = [[x0,y0]]
        points.extend([points2[j] for j in range(pointNum)])
        points.append([x0,y0])
        #points.append((x0,y0))
    else:
        rads = np.random.uniform(smallestSize+minRad, 1-smallestSize, pointNum)
        points = [(rads[val]*np.cos(phis[val]),rads[val]*np.sin(phis[val])) for val in range(pointNum)]
        points.insert(0,(0,0))
        points.append((x0,y0))
    return np.array(points)