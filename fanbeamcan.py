# This is a python script that translates, the straight line transform,
# WF of image to WF of sinogram space in fan beam coordinates.
#The input will be ImageWF, which is an NxNx180 tensor of zeros and ones.


#import tensorflow as tf
import numpy as np
import math
import torch

# Here we create some ImageWF tensor to check our code
# Set resolution N on image space
# Must choose an even number
N = 201
# Set WF on image space
# We set it to zeros for the time being but we will use other inputs
ImageWF = np.zeros([N, N, 180])

#this block sets some test images for sanity check
j=0
while j <= N-1:
    ImageWF[j, 100, 0] = 1
    j = j+1


#Actual code for WF mapping starts here

#initializing sinogram WF
SinoWF = np.zeros([N,180,180])
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
SinoWFtensor = torch.tensor(SinoWF)
print(torch.nonzero(SinoWFtensor))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
