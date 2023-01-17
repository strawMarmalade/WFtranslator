import numpy as np
import matplotlib.pyplot as plt
import warnings
import io
import time
from matplotlib.patches import Ellipse

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
        
def rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], 
                         [np.sin(theta),  np.cos(theta)]])

def genEll():
    angle = np.random.uniform(-1,1)*np.pi/4
    
    maxlen = np.sqrt(1+(np.tan(angle))**2)
    a = np.random.uniform(0,maxlen)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        maxb1 = np.sqrt((1-(a*np.cos(angle))**2)/(np.sin(angle)**2))
        maxb2 = np.sqrt((1-(a*np.sin(angle))**2)/(np.cos(angle)**2))
    
    maxb = np.min([maxb1,maxb2])

    b = np.random.uniform(0,maxb)
    
    #these are the maximal distances to border
    maxX = 1-np.sqrt((a*np.cos(angle))**2+(b*np.sin(angle))**2)
    maxY = 1-np.sqrt((a*np.sin(angle))**2+(b*np.cos(angle))**2)
        
    centerX = 0#np.random.uniform(-maxX,maxX)
    centerY = 0#np.random.uniform(-maxY,maxY)
    
    center = np.array([centerX,centerY])
    return Ellipse(center, 2*a,2*b, angle=np.rad2deg(angle))


# fig = plt.figure(figsize=(1,1), dpi=200, frameon=False)
# plt.xlim(-1,1)
# plt.ylim(-1,1)
# plt.axis('off')
# fig.add_artist(e)
# plt.show()

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
    return grid

def plotEll(ell, stepSize=200, output=True):
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    ax.add_artist(ell)
    plt.show()
    


#maybe wont work, i dunno
def checkIfPointInEllipse(p, ell):
    center, angle, a, b = ell
    pRotatedBack = rot(-angle)@(p-center)
    if -a <= pRotatedBack[0] <= a:
        if -b*np.sqrt(1-((pRotatedBack[0]/a))**2) <= pRotatedBack[1] <= b*np.sqrt(1-((pRotatedBack[0]/a))**2):
            return True
    return False


def drawGrid(grid):
    plt.imshow(grid, origin='lower') 
    plt.show()


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

np.random.seed(107)

def calc2plus(a,b,p,x):
    return (np.sin(p)*np.cos(p)*(b**2-a**2)-2*x*a*b/np.sqrt((a*np.cos(p))**2+(b*np.sin(p))**2-x**2))/((a*np.cos(p))**2+(b*np.sin(p))**2)

def calc2minus(a,b,p,x):
    return (np.sin(p)*np.cos(p)*(b**2-a**2)+2*x*a*b/np.sqrt((a*np.cos(p))**2+(b*np.sin(p))**2-x**2))/((a*np.cos(p))**2+(b*np.sin(p))**2)


def calc(a,b,p):
    #return  (a*np.cos(t)*np.sin(p)+b*np.sin(t)*np.cos(p) +y0)/(a*np.cos(t)*np.cos(p)-b*np.sin(t)*np.sin(p)+x0)*(a**2*np.cos(p)-b**2*np.sin(p))/(b**2*np.cos(p)+a**2*np.sin(p))
    return ((a**2)*np.cos(p)-(b**2)*np.sin(p))/((b**2)*np.cos(p)+(a**2)*np.sin(p))

def ellipseToWFsetList(ell,gridSize=200, angleAccuracy=360, method=4):
    a = ell.get_width()/2
    b = ell.get_height()/2
    x0,y0 = ell.get_center()
    angle = np.deg2rad(ell.get_angle())

    if method == 2:
        t = np.linspace(-np.pi/2, 3/2*np.pi, angleAccuracy)
        Ellrot = np.array([a*np.cos(t)*np.cos(angle)-b*np.sin(t)*np.sin(angle)+x0, a*np.cos(t)*np.sin(angle)+b*np.sin(t)*np.cos(angle)+y0])  

        anglesOther2 = [360+np.rad2deg(angle+np.arctan2(np.tan(j)*a,b)) for j in np.linspace(-np.pi/2,np.pi/2, angleAccuracy//2)]
        anglesOther2.extend([360+np.rad2deg(np.pi+angle+np.arctan(np.tan(j)*a/b)) for j in np.linspace(np.pi/2, 3/2*np.pi, angleAccuracy//2)])
        anglesOther2[180] += 180
        return [[point2grid(np.array([Ellrot[0,j],Ellrot[1,j]]),gridSize=gridSize),[anglesOther2[j]]] for j in range(angleAccuracy)]

    else:
        t = np.linspace(0, 2*np.pi, angleAccuracy)
        Ellrot = np.array([a*np.cos(t)*np.cos(angle)-b*np.sin(t)*np.sin(angle)+x0, a*np.cos(t)*np.sin(angle)+b*np.sin(t)*np.cos(angle)+y0])  

        angle5 = [np.rad2deg(3/2*np.pi+np.arctan2(Ellrot[1,j+1]-Ellrot[1,j],Ellrot[0,j+1]-Ellrot[0,j])) for j in range(angleAccuracy-1)]
        angle5.extend([np.rad2deg(3/2*np.pi+np.arctan2(Ellrot[1,-1]-Ellrot[1,-2],Ellrot[0,-1]-Ellrot[0,-2]))])
        return [[point2grid(np.array([Ellrot[0,j],Ellrot[1,j]]),gridSize=gridSize),[angle5[j]]] for j in range(angleAccuracy)]

    
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
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     #angle3 = [np.rad2deg(np.arctan(Ellrot[1,j]/Ellrot[0,j]*calc(a,b,angle))) for j in range(angleAccuracy//2)]
    #     #angle3.extend([np.rad2deg(np.pi+np.arctan(Ellrot[1,j]/Ellrot[0,j]*calc(a,b,angle))) for j in range(angleAccuracy//2,angleAccuracy)])

    #     #anglesPlus = [j+np.rad2deg(angle) for j in range(angleAccuracy)]
    #     anglesOther2 = [np.rad2deg(angle+np.arctan2(np.tan(j)*a,b)) for j in np.linspace(-np.pi/2,np.pi/2, angleAccuracy//2)]
    #     anglesOther2.extend([np.rad2deg(np.pi+angle+np.arctan(np.tan(j)*a/b)) for j in np.linspace(np.pi/2, 3/2*np.pi, angleAccuracy//2)])

    #     #
    #     #anglesOther = [np.rad2deg(angle+np.arctan2(Ell[1,j]/(b**2),Ell[0,j]/(a**2))) for j in range(angleAccuracy)]
    #     #anglesPlus = [np.round((angle+np.arctan2(1,2*b*(a**(-2))*Ell[0,j]/(np.sqrt(1-(Ell[0,j]/a)**2))))*angleAccuracy/(2*np.pi)).astype(int)%angleAccuracy for j in range(angleAccuracy//2)]
    #     #anglesPlus.extend([np.round((angle+np.arctan2(-1,2*b*(a**(-2))*Ell[0,j]/(np.sqrt(1-(Ell[0,j]/a)**2))))*angleAccuracy/(2*np.pi)).astype(int)%angleAccuracy for j in range(angleAccuracy//2,angleAccuracy)])
    
    #     #angle4 = [np.rad2deg(np.arctan2(-1,calc2minus(a,b,angle, Ellrot[0,j]))) for j in range(angleAccuracy//2)]
    #     #angle4.extend([np.rad2deg(np.arctan2(-1,calc2minus(a,b,angle, Ellrot[0,j]))) for j in range(angleAccuracy//2,angleAccuracy)])
    
    # angle5 = [np.rad2deg(3/2*np.pi+np.arctan2(Ellrot[1,j+1]-Ellrot[1,j],Ellrot[0,j+1]-Ellrot[0,j])) for j in range(angleAccuracy-1)]
    # angle5.extend([np.rad2deg(3/2*np.pi+np.arctan2(Ellrot[1,-1]-Ellrot[1,-2],Ellrot[0,-1]-Ellrot[0,-2]))])
    # #angle5.extend([np.rad2deg(3/2*np.pi+np.arctan((Ellrot[1,(j+1)%360]-Ellrot[1,j%360])/(Ellrot[0,(j+1)%360]-Ellrot[0,j%360]))) for j in range(angleAccuracy//2, angleAccuracy)])

    # #anglediff = [angle5[j]-anglesOther2[j] for j in range(angleAccuracy)]
    # # xnew = [x0+val for val in x]
    
    
    # #method 1 is rubbish
    # #method 3 is rubbish too
    # if method == 3:
    #     WFSetList = [[point2grid(np.array([Ellrot[0,j],Ellrot[1,j]]),gridSize=gridSize),[angle3[j]]] for j in range(angleAccuracy)]
    # elif method == 2:
    #     WFSetList = [[point2grid(np.array([Ellrot[0,j],Ellrot[1,j]]),gridSize=gridSize),[anglesOther2[j]]] for j in range(angleAccuracy)]
    # elif method == 1:
    #     WFSetList = [[point2grid(np.array([Ellrot[0,j],Ellrot[1,j]]),gridSize=gridSize),[anglesPlus[j]]] for j in range(angleAccuracy)]
    # elif method == 4:
    #     WFSetList = [[point2grid(np.array([Ellrot[0,j],Ellrot[1,j]]),gridSize=gridSize),[angle5[j]]] for j in range(angleAccuracy)]

    # #WFSetList.extend([[point2grid(np.array([x0+Ell_rot[0,j],y0+Ell_rot[1,j]])),[anglesMinus[val]]] for j in range(angleAccuracy//2,angleAccuracy)]
    
    # return WFSetList

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
            stepTakenAsGrid = np.array(pointToGridIndex(stepTaken[0], stepTaken[1], gridSize))
           # WFSetGrid[stepTakenAsGrid[0]][stepTakenAsGrid[1]][outwardWFEndAngle] = 1
            WFSetList.append([stepTakenAsGrid,[outwardWFEndAngle]])
        #the following list will be filled with every angle between the two outward pointing
        #angles from above which is the set of outward wavefront directions for a corner point
        #of a polygon     
        
        pointAsGrid = np.array(pointToGridIndex(poly[val][0],poly[val][1],gridSize))
        
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
             
def drawWFSetList(WFSetList,gridSize=200, saveFile=True, method=1):
    #fig = plt.figure(figsize=(2,2), dpi=900)
    for val in range(len(WFSetList)):
        pointGrid = WFSetList[val][0]
        point = grid2point(pointGrid,gridSize)
        angles = WFSetList[val][1]
        for angle in angles:
            #to plot the WFset we just make small lines in the correct direction
            #at the point
            vec = [0.05*np.cos((2*np.pi*angle/360)), 0.05*np.sin((2*np.pi*angle/360))]
            plt.plot([point[0],point[0]+vec[0]],[point[1],point[1]+vec[1]],color='black',linewidth=0.3)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    if saveFile:
        plt.savefig(f'fileAngle{method:d}.png',dpi=900)
    plt.show()

def convertWFListToWFGridLeoConvetion(List, gridSize=200, angleAccuracy=360):
    LeoAngleAcc = int(angleAccuracy/2)
    WFSetGrid = np.zeros([gridSize,gridSize, int(LeoAngleAcc)])
    for val in List:
        point = val[0]
        angleListHalf = [ang%LeoAngleAcc for ang in val[1]]
        WFSetGrid[point[0],point[1], angleListHalf] = 1
    return WFSetGrid

def newCheckIfInTriangle(p,xs):
    a = 1/2 * (-xs[1,1] * xs[2,0] + xs[0,1] * (-xs[1,0] + xs[2,0]) + xs[0,0] * (xs[1,1] - xs[2,1]) + xs[1,0] * xs[2,1])
    s = (xs[0,1]*xs[2,0] - xs[0,0]*xs[2,1] + (xs[2,1] - xs[0,1])*p[0] + (xs[0,0] - xs[2,0])*p[1])
    t = (xs[0,0]*xs[1,1] - xs[0,1]*xs[1,0] + (xs[0,1] - xs[1,1])*p[0] + (xs[1,0] - xs[0,0])*p[1])

    return s > 0 and t > 0 and (s + t) < 2 * a

def checkIfPointIsInPolygonNew2(p, poly):
    #as the first and last element in the list poly is the same starting point, we leave it away
    for index in range(1,len(poly)-2):
        xs = np.array([poly[0],poly[index],poly[index+1],poly[0]])
        if np.min(xs[:,0]) <= p[0] <= np.max(xs[:,0]) and np.min(xs[:,1]) <= p[1] <= np.max(xs[:,1]):
        #the only polgons we can construct are star-convex so we just have to check for
        #all the triangles that make up the polygon
            if newCheckIfInTriangle(p, xs):
                return True
    return False

def gridFromPolygon(poly,gridSize=200):
    dpi = int(gridSize/2)
    fig = plt.figure(figsize=(2,2), dpi=dpi, frameon=False)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.axis('off')
    plt.fill(poly[:,0],poly[:,1], "black")
    io_buf = io.BytesIO()
    fig.savefig(io_buf,format='raw', dpi=dpi)
    io_buf.seek(0)
    grid6 = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                          newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))[:,:,0] < 255
    io_buf.close()
    plt.close(fig)
    return np.transpose([np.flip(grid6[:,val]) for val in range(gridSize)])


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

    tic = time.perf_counter()
    grid = gridFromPolygon(poly, gridSize=gridSize)
    toc = time.perf_counter()
    print(f"Get inside of polygon way as grid took {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    drawGrid(grid)
    toc = time.perf_counter()
    print(f"Drawing the grid of polygon took {toc - tic:0.4f} seconds\n")

def fullEllipseRoutineTimer(gridSize = 800, angleAccuracy=360):   
    print(f"Grid size is {gridSize:d}")

    tic = time.perf_counter()
    ell = genEll()
    toc = time.perf_counter()
    print(f"Ellipse generation took {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    WFSetList2 = ellipseToWFsetList(ell, gridSize=gridSize, angleAccuracy=360, method=2)
    drawWFSetList(WFSetList2, gridSize=gridSize, saveFile=True,method=2)
    toc = time.perf_counter()
    print(f"Wavefrontset calculation took {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    WFSetList3 = ellipseToWFsetList(ell, gridSize=gridSize, angleAccuracy=360, method=4)
    drawWFSetList(WFSetList3, gridSize=gridSize, saveFile=True,method=4)
    #plotEll(ell, output=False)
    toc = time.perf_counter()
    print(f"Plotting ellipse with wavefrontset picture took {toc - tic:0.4f} seconds")
    
    tic = time.perf_counter()
    grid = gridEll(ell, gridSize=gridSize)
    toc = time.perf_counter()
    print(f"Get inside of ellipse as grid took {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    # plt.imshow(grid)
    # plt.show()
    toc = time.perf_counter()
    print(f"Drawing the grid of polygon took {toc - tic:0.4f} seconds\n")

#fullPolygonRoutineTimer(polySize=5)
fullEllipseRoutineTimer()


# ell = generateEllipse()
# drawEllipse(ell, output=False)
# grid = constructImageInGridOfEllipse(ell)

# print(radonTrafo(grid))


# tic = time.perf_counter()
# grid1 = polyGrid1(poly)
# toc = time.perf_counter()
# print(f"Way 1 took {toc - tic:0.4f} seconds\n")
# tic = time.perf_counter()
# grid2 = polyGrid2(poly)
# toc = time.perf_counter()
# print(f"Way 2 took {toc - tic:0.4f} seconds\n")
# tic = time.perf_counter()
# grid3 = polyGrid3(poly)
# toc = time.perf_counter()
# print(f"Way 3 took {toc - tic:0.4f} seconds\n")
# tic = time.perf_counter()
# grid4 = gridOfAllPointsInPolygon(poly, gridSize=200, angleAccuracy=360)
# toc = time.perf_counter()
# print(f"Old way took {toc - tic:0.4f} seconds\n")
# tic = time.perf_counter()
# grid5 = gridOfAllPointsInPolygon2(poly, gridSize=200, angleAccuracy=360)
# toc = time.perf_counter()
# print(f"Old new way took {toc - tic:0.4f} seconds\n")
# tic = time.perf_counter()
# fig = plt.figure(figsize=(2,2), dpi=100, frameon=False)
# plt.xlim(-1,1)
# plt.ylim(-1,1)
# plt.axis('off')
# plt.fill(poly[:,0],poly[:,1], "black")
# io_buf = io.BytesIO()
# fig.savefig(io_buf,format='raw', dpi=100)
# io_buf.seek(0)
# grid = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
#                       newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))[:,:,0] < 255
# grid = np.transpose([np.flip(grid[:,val]) for val in range(200)])
# io_buf.close()
# plt.close(fig)
# plt.imshow(grid, origin='lower')
# toc = time.perf_counter()
# print(f"Matplotlib took {toc - tic:0.4f} seconds\n")


#drawGrid(grid)
# drawGrid(grid2)
# drawGrid(grid3)
# drawGrid(grid4)
# drawGrid(grid5)




# nums = 20
# way1 = 0.0
# way2 = 0.0
# way3 = 0.0
# way4 = 0.0
# way5 = 0.0
# way6 = 0.0
# size = 1000
# dp = int(size/2)
# for k in range(nums):
#     poly = generatePolygon(7)
#     # tic = time.perf_counter()
#     # grid1 = polyGrid1(poly)
#     # toc = time.perf_counter()
#     # way1 += toc - tic
#     # print(f"Way 1 took {toc - tic:0.4f} seconds\n")
#     # tic = time.perf_counter()
#     # grid2 = polyGrid2(poly)
#     # toc = time.perf_counter()
#     # way2 += toc - tic
#     # print(f"Way 2 took {toc - tic:0.4f} seconds\n")
#     # tic = time.perf_counter()
#     # grid3 = polyGrid3(poly)
#     # toc = time.perf_counter()
#     # way3 += toc - tic
#     # print(f"Way 3 took {toc - tic:0.4f} seconds\n")
#     # tic = time.perf_counter()
#     # grid4 = gridOfAllPointsInPolygon(poly, gridSize=size, angleAccuracy=360)
#     # toc = time.perf_counter()
#     # way4 += toc - tic
#     # print(f"Old way took {toc - tic:0.4f} seconds\n")
#     tic = time.perf_counter()
#     grid5 = gridOfAllPointsInPolygon2(poly, gridSize=size, angleAccuracy=360)
#     toc = time.perf_counter()
#     way5 += toc - tic
#     print(f"Old new way took {toc - tic:0.4f} seconds\n")
#     tic = time.perf_counter()
#     fig = plt.figure(figsize=(2,2), dpi=dp, frameon=False)
#     plt.xlim(-1,1)
#     plt.ylim(-1,1)
#     plt.axis('off')
#     plt.fill(poly[:,0],poly[:,1], "black")
#     io_buf = io.BytesIO()
#     fig.savefig(io_buf,format='raw', dpi=dp)
#     io_buf.seek(0)
#     grid6 = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
#                           newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))[:,:,0] < 255
#     io_buf.close()
#     grid6 = np.transpose([np.flip(grid6[:,val]) for val in range(200)])
#     plt.close(fig)
#     toc = time.perf_counter()
#     way6 += toc - tic
#     print(f"Pyplot took {toc - tic:0.4f} seconds\n")

# way1 /= nums
# way2 /= nums
# way3 /= nums
# way4 /= nums
# way5 /= nums
# way6 /= nums


# print(f"Way 1: {way1:0.4f}\nWay 2: {way2:0.4f}\nWay 3: {way3:0.4f}\nWay 4: {way4:0.4f}\nWay 5: {way5:0.4f}\nWay 6: {way6:0.4f}")

# Way 1: 2.7858
# Way 2: 1.9747
# Way 3: 2.0195
# Way 4: 0.5225
# Way 5: 0.4775
# Way 6: 0.0105

#so way 4 is the way to go
# Way 1: 0.8512
# Way 2: 0.5118
# Way 3: 0.5232
# Way 4: 0.3255
# Way 5: 0.2634

#size 5:
# Way 1: 0.7245
#  Way 2: 0.4681
#  Way 3: 0.4778
#  Way 4: 1.1108


# ### Leo stuff
# import math
# import torch
# # N = 201

# # ImageWF = convertWFListToWFGridLeoConvetion(List, gridSize=201)
# # #ImageWF = np.zeros([N, N, 180])

# # #this block sets some test images for sanity check
# # # j=0
# # # while j <= N-1:
# # #     ImageWF[j, 1, 0] = 1
# # #     j = j+1


# # #Actual code for WF mapping starts here

# # #initializing sinogram WF
# # SinoWF = np.zeros([N,180,180])
# def canonicalplus1(r, alpha, phi):
#     return (math.acos(r*math.cos(alpha - phi)) + phi)
# def canonicalminus1(r, alpha, phi):
#     return (-math.acos(r*math.cos(alpha - phi)) + phi)
# def canonicalplus2(r, alpha, phi):
#     return -math.asin(r*math.cos(alpha-phi)/2)
# def canonicalminus2(r, alpha, phi):
#     return math.asin(r*math.cos(alpha-phi)/2)
# def traveltimeplus(r, alpha, phi):
#     return math.sqrt(4 -(r* math.cos(alpha - phi))**2) - r*math.sin(alpha-phi)
# def traveltimeminus(r, alpha, phi):
#     return math.sqrt(4 -(r* math.cos(alpha - phi))**2) + r*math.sin(alpha-phi)
# def pullback(rho, theta, phi, t):
#     JMatrix = np.array([[-2* math.sin(rho) + t* math.sin(theta + rho), 2*math.cos(rho) - t* math.cos(theta+rho)],[ t*math.sin(theta+rho), -t*math.cos(theta+rho)]])
#     ImageWFVect = np.array([math.cos(phi), math.sin(phi)])
#     SinoWFVect = JMatrix.dot(ImageWFVect)
#     #print(JMatrix)
#     #print(SinoWFVect)
# #    print(math.degrees(math.acos(SinoWFVect[0]/math.sqrt(SinoWFVect[0]**2 + SinoWFVect[1]**2))))
#     return round(math.degrees(math.acos(SinoWFVect[0]/math.sqrt(SinoWFVect[0]**2 + SinoWFVect[1]**2))))




# # rowindex = 0
# # while (rowindex <= N-1):
# #     colindex = 0
# #     while (colindex <= N-1):
# #         WFangleindex = 0
# #         while (WFangleindex <= 179):
# #             if ImageWF[rowindex, colindex, WFangleindex] ==1:
# #                radius = math.sqrt((2*rowindex/(N-1) -1)**2 + (2*colindex/(N-1) - 1)**2)
# #                #computes the distance of the pixel from the origin
# #                if radius ==0:
# #                    positionangle = 0
# #                else:
# #                    #print((2 * colindex / (N - 1) - 1) / radius)
# #                    #print(math.acos((2 * colindex / (N - 1) - 1) / radius))
# #                    positionangle = np.sign((2*rowindex/(N-1) -1))*math.acos((2 * colindex / (N - 1) - 1) / radius)
# #                    if np.sign((2*rowindex/(N-1) -1))== 0:
# #                        positionangle = math.acos((2 * colindex / (N - 1) - 1) / radius)
# #                    #print(positionangle)
# #                #positionangle is the angle of the position measured in Radians. It takes the range between -pi to pi
# #                WFangleradian = math.radians(WFangleindex)
# #                #turns WFangle from entry index to radians
# #                boundaryradplus = canonicalplus1(radius, positionangle, WFangleradian)
# #                #above function returns location on the boundary of circle in radians.
# #                # So the range is a float between 0 and 2pi
# #                boundarydegreeplus = math.degrees(boundaryradplus)
# #                boundaryindexplus = round(boundarydegreeplus *N/360)%N
# #                boundaryradminus = canonicalminus1(radius, positionangle, WFangleradian)
# #                boundarydegreeminus = math.degrees(boundaryradminus)
# #                boundaryindexminus = round(boundarydegreeminus *N/360)%N
# #                incomingradplus = canonicalplus2(radius, positionangle, WFangleradian)
# #                #above function returns incoming direction in radians relative to the inward pointing normal
# #                #so the range is an integer between -pi/2 degrees to pi/2 degrees
# #                incomingdegreeplus = math.degrees(incomingradplus)
# #                incomingradminus = - incomingradplus
# #                incomingdegreeminus = -incomingdegreeplus
# #                incomingindexplus = round(incomingdegreeplus + 90)%180
# #                incomingindexminus = round(incomingdegreeminus + 90)%180
# #                tplus = traveltimeplus(radius, positionangle, WFangleradian)
# #                tminus = traveltimeminus(radius, positionangle, WFangleradian)
# #                SinoWFindexplus = pullback(boundaryradplus, incomingradplus, WFangleradian, tplus)
# #                SinoWFindexminus = pullback(boundaryradminus, incomingradminus, WFangleradian, tminus)
# #                SinoWF[boundaryindexplus, incomingindexplus, SinoWFindexplus] = 1
# #                SinoWF[boundaryindexminus, incomingindexminus, SinoWFindexminus] = 1
# #             WFangleindex = WFangleindex + 1
# #         colindex = colindex + 1
# #     rowindex = rowindex + 1        

# # print(SinoWF)
# # SinoWFtensor = torch.tensor(SinoWF)
# # print(torch.nonzero(SinoWFtensor))
    