# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:42:03 2019

@author: ADEJUNIOR
"""

import cv2
import svgwrite
import numpy as np
import math
from sklearn.decomposition import PCA
from PIL import Image
from skimage.morphology import skeletonize
# from skimage.filters import (threshold_otsu, threshold_niblack,
#                              threshold_sauvola)

import colorsys
import matplotlib.pyplot as plt
from numba import jit

# from skimage.data import binary_blobs
# import scipy.stats as st
# import pycircstat as cs
# import datetime

theta = math.pi/180


def show_image(image):
    '''
    Opens image on system's viewer.

    Parameters
    ----------
    image : TYPE image
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    pil_img = Image.fromarray(image)
    pil_img.show()
    return


def auto_canny(image, sigma=0.33):
    '''
    Apply automatic Canny edge detection using the computed median
    https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    sigma : TYPE, optional
        DESCRIPTION. The default is 0.33.

    Returns
    -------
    edged : TYPE
        DESCRIPTION.

    '''

    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged


@jit(nopython=True)
def adaptative_thresholding(image, kernelsize, mode):
    '''
    Apply a adaptative thresholding method to a grayscale image array

    Parameters
    ----------
    image : TYPE np.uint8
        Grayscale image.
    kernelsize : TYPE int
        Kernel size.
    mode : TYPE string
        Method name e.g. "sauvola", "niblack", "phansalkar".

    Returns
    -------
    newimage : TYPE np.uint8
        Grayscale image.

    '''
    if kernelsize % 2 == 0:  # Exit function if kernel is even
        return

    offset = int(kernelsize/2)  # Pixels for each side around a kernel center

    new_image = np.zeros(np.shape(image))  # Thresholded image

    for i in range(0, np.shape(image)[1]):  # collumn
        for j in range(0, np.shape(image)[0]):  # row

            # Create and change offsets for each direction according to (i,j)
            # position near the limits of the image
            offsetL = offsetR = offsetU = offsetD = offset
            if j <= offset:
                offsetL = offset + (j - offset)
            if np.shape(image)[0] - j <= offset:
                offsetR = np.shape(image)[0] - j
            if i <= offset:
                offsetU = offset + (i - offset)
            if np.shape(image)[1] - i <= offset:
                offsetD = np.shape(image)[1] - i

            # Store in memory (temp) the kernel for (i,j) position
            temp = image[j-offsetL:j+offsetR+1, i-offsetU:i+offsetD+1]

            # Apply thresholding method
            if mode == 'niblack':
                T = int(np.mean(temp) + 0.2*np.std(temp))
                if image[j, i] <= T:
                    new_image[j, i] = 0
                else:
                    new_image[j, i] = 255
            if mode == 'sauvola':
                T = int(np.mean(temp)*(1 + 0.5*(np.std(temp)/128 - 1)))
                if image[j, i] <= T:
                    new_image[j, i] = 0
                else:
                    new_image[j, i] = 255
    return new_image


def skeletonize_image(image, invert, mode):
    '''
    Apply morphologic operation skeletonize

    Parameters
    ----------
    image : Array of uint8
        Thresholded image with 0 and 255 values.
    invert : bool
        Invert the threshold elements in image.
    mode : str
        Method of skeletonization.

    Returns
    -------
    new_image : TYPE
        DESCRIPTION.

    '''
    if invert:
        new_image = np.zeros(np.shape(image))
        for i in range(0, np.shape(image)[1]):
            for j in range(0, np.shape(image)[0]):
                if image[j, i] == 255:
                    new_image[j, i] = 0
                else:
                    new_image[j, i] = 255

    new_image = cv2.threshold(new_image, 100, 1, cv2.THRESH_BINARY)[1]
    new_image = skeletonize(new_image)*255
    return new_image


@jit(nopython=True)
def bresenham_march(image, p1, p2):
    '''
    Line iterator substituting opencv removed implementation
    https://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator

    Parameters
    ----------
    image : Array of uint8
        Grayscale image.
    p1 : list
        x and y coordinates of first point.
    p2 : list
        x and y coordinates of first point.

    Returns
    -------
    ret : list
        List of pixel intensities for the given segment.

    '''
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    # tests if any coordinate is outside the image
    if (
        x1 >= image.shape[0]
        or x2 >= image.shape[0]
        or y1 >= image.shape[1]
        or y2 >= image.shape[1]
    ):  # tests if line is in image, necessary because some part of the line
        # must be inside, it respects the case that the two points are outside
        if not cv2.clipLine((0, 0, image.shape[0], image.shape[1]), p1, p2):
            print("not in region")
            return

    steep = math.fabs(y2 - y1) > math.fabs(x2 - x1)
    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # takes left to right
    also_steep = x1 > x2
    if also_steep:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    dx = x2 - x1
    dy = math.fabs(y2 - y1)
    error = 0.0
    delta_error = 0.0
    # Default if dx is zero
    if dx != 0:
        delta_error = math.fabs(dy / dx)

    y_step = 1 if y1 < y2 else -1

    y = y1
    ret = []
    for x in range(x1, x2):
        p = (y, x) if steep else (x, y)
        if p[0] < image.shape[0] and p[1] < image.shape[1]:
            # ret.append((p, image[p]))
            ret.append(image[p])
        error += delta_error
        if error >= 0.5:
            y += y_step
            error -= 1
    if also_steep:  # because we took the left to right instead
        ret.reverse()
    return ret





##########################################################################    
#Connect lines that end close to each other adding new lines between them
def connectLines(image, lines, angles, window, alpha_limit, beta_limit, mode):
    n = np.shape(lines)[0] #Number of lines
    connections = np.zeros((n,n), dtype = bool) #Matrix of connections between all segments, starting with false
    count = np.zeros((n), dtype = np.int)
    new_lines = [] #List of new segments to be added later
    
    ##########################################################################    
    #Given a reference point(vertice), finds anther close segments
    def checkAround2(lines, vertice, index, window): # O calculo do ângulo entre duas retas precisa ser atuaizado
        #if window % 2 == 0: # Transforms the window to a odd size
        #    window = window + 1
        #offset = int(window/2) # This is used to centralize the clipline window
        offset = window
        segm = list()
        for j in range(0,n):
            if j != index and count[j] != 2:
                if checkConnection(connections, index, j):
                    return False
                dist1 = compute_distance(vertice[0], vertice[1], lines[j][0], lines[j][1])
                dist2 = compute_distance(vertice[0], vertice[1], lines[j][2], lines[j][3])
                aux = False
                if dist1 <= offset or dist2 <= offset:
                    aux = True
                #aux = cv2.clipLine((vertice[0]-offset, vertice[1]-offset, window, window), (lines[j][0],lines[j][1]), (lines[j][2],lines[j][3]))
                if aux != False:
                    segm.append(j) # Save the lines indexes
        if np.size(segm) == 0:
            return False
        else:
            link = []
            for k in segm:
                alpha = compare_angles(vertice, lines[index], lines[k])
                if alpha[0] >= alpha_limit: # and alpha[1] > 1
                    new_segm = [int(vertice[0]), int(vertice[1]), int(alpha[2]), int(alpha[3])]
                    beta = compare_angles(vertice, lines[index], new_segm)
                    if beta[0] >= beta_limit:
                        length = sd = 0
                        new_segm.append(alpha[0])
                        new_segm.append(beta[0])
                        if mode == 'distance':
                            length = compute_distance(new_segm[0], new_segm[1], new_segm[2], new_segm[3])
                        new_segm.append(length)
                        if mode == 'deviation':
                            pixels = bresenham_march(image, (new_segm[0],new_segm[1]), (new_segm[2],new_segm[3]))
                            #print(pixels)
                            sd = np.median(pixels)# + np.std(pixels)
                        new_segm.append(sd)
                        new_segm.append(k)
                        link.append(new_segm)               
            if np.size(link) != 0:
                link = np.asarray(link).flatten().reshape((int(np.size(link)/9),9))
                if mode == 'alpha': # greater alpha
                    row = np.where(link[:,4] == np.amax(link[:,4]))[0][0]
                if mode == 'beta': # greater beta
                    row = np.where(link[:,5] == np.amax(link[:,5]))[0][0]
                if mode == 'distance': # minor distance
                    row = np.where(link[:,6] == np.amin(link[:,6]))[0][0]
                if mode == 'deviation': # minor deviation
                    row = np.where(link[:,6] == np.amin(link[:,6]))[0][0]
                    
                
                return (link[row][0], link[row][1], link[row][2], link[row][3], link[row][8])
                
                '''
                pixels = bresenham_march(image, (int(link[row][0]), int(link[row][1])), (int(link[row][2]), int(link[row][3])))    
                
                if np.size(pixels) == 0:
                    return (link[row][0], link[row][1], link[row][2], link[row][3], link[row][8]) 
                if np.size(pixels) == 1:
                    ends = pixels[0]    
                else:
                    plt.plot(pixels)
                    plt.show()
                    ends = ((pixels[0]+pixels[-1])/2) + 10
                    print(np.mean(pixels), (pixels[0]+pixels[-1])/2, np.var(pixels))
                 
                 
                if np.size(pixels) > 1:
                    plt.plot(pixels)
                    plt.show()
                    print(np.mean(pixels), (pixels[0]+pixels[-1])/2, np.var(pixels))
                    
                if np.var(pixels) < 200:
                    #print(np.size(pixels), np.mean(pixels), np.median(pixels), np.std(pixels))
                    #print(np.mean(pixels), (pixels[0]+pixels[-1])/2, np.var(pixels))
                    #plt.plot(pixels)
                    #plt.show()
                    return (link[row][0], link[row][1], link[row][2], link[row][3], link[row][8])
                else:
                    return False '''
            else:
                return False
            

        
    
    ##########################################################################    
    #Check if a connection exists between two lines. If is True in the matrix of connections
    @jit(nopython=True)
    def checkConnection(connections, line1, line2):
        index = [line1, line2]
        index.sort()
        return connections[index[0]][index[1]]
    
    
    ##########################################################################    
    #Make a connection in the matrix of connections True
    @jit(nopython=True)
    def addConnection(connections, line1, line2):
        index = [line1, line2]
        index.sort()
        connections[index[0]][index[1]] = True
        return connections
    

    ##########################################################################    
    #Create segments between two close segments
    def addSegments(index, side, new_lines):
        vertice = list((lines[i][0+side*2],lines[i][1+side*2]))
        segm = checkAround2(lines, vertice, index, window)
        if type(segm) == tuple:
            aux = list((segm[0], segm[1], segm[2], segm[3]))
            addConnection(connections, index, int(segm[4]))
            count[index] += 1
            new_lines = np.insert(new_lines, np.shape(new_lines)[0] ,aux, axis = 0)
            return new_lines
        else:
            return new_lines   
    
    
    #Search for connections and save the new segments
    for i in range(0,n):
        new_lines = addSegments(i, 0, new_lines)
        new_lines = addSegments(i, 1, new_lines)
    
    #Reshape matrix  
    new_lines = np.reshape(new_lines, (np.uint64(np.shape(new_lines)[0]/4),4))
    #lines_0 = lines
    #Add the aditional segments to the list of segments
    return np.insert(lines, np.shape(lines)[0], new_lines, axis = 0)


def compute_distance(x0, y0, x1, y1):
    '''
    Compute the distance between two point p and q

    Parameters
    ----------
    x0 : TYPE
        DESCRIPTION.
    y0 : TYPE
        DESCRIPTION.
    x1 : TYPE
        DESCRIPTION.
    y1 : TYPE
        DESCRIPTION.

    Returns
    -------
    dist : float
        Distance in pixel size.

    '''
    dist = math.sqrt(math.pow((x1 - x0), 2) + math.pow((y1 - y0), 2))
    return dist


def compare_angles(base, line1, line2):
    x0, y0 = base
    if [line1[0], line1[1]] == [x0, y0]:
        px, py = [line1[2], line1[3]]
    else:
        px, py = [line1[0], line1[1]]
        
    dist0 = compute_distance(x0,y0,line2[0],line2[1])
    dist1 = compute_distance(x0,y0,line2[2],line2[3])
    
    if dist0 < dist1:
        px = px - x0
        py = py - y0
        qx, qy = [line2[2] - x0, line2[3] - y0]
        aux = ((px*qx) + (py*qy))/(math.sqrt(math.pow(px,2) + math.pow(py,2))*math.sqrt(math.pow(qx,2) + math.pow(qy,2)))
        try:
            angle = math.acos(aux)*180/math.pi
        except:
            if (aux) > 0:
                angle = 0
            if (aux) < 0:
                angle = 180
        return (angle, dist0, line2[0], line2[1])
    else:
        px = px - x0
        py = py - y0
        qx, qy = [line2[0] - x0, line2[1] - y0]
        aux = ((px*qx) + (py*qy))/(math.sqrt(math.pow(px,2) + math.pow(py,2))*math.sqrt(math.pow(qx,2) + math.pow(qy,2)))
        try:
            angle = math.acos(aux)*180/math.pi
        except:
            if (aux) > 0:
                angle = 0
            if (aux) < 0:
                angle = 180
        return (angle, dist1, line2[2], line2[3])    


def findIntersection(line1, line2):
    x0, y0, x1, y1 = line1
    x2, y2, x3, y3 = line2
    
    if dointersect(point(x0,y0),point(x1,y1), point(x2,y2), point(x3,y3)) == False:
        return False
    
    try:
        denom = float((x0 - x1) * (y2 - y3) - (y0 - y1) * (x2 - x3))
        x = ((x0 * y1 - y0 * x1) * (x2 - x3) - (x0 - x1) * (x2 * y3 - y2 * x3)) / denom
        y = ((x0 * y1 - y0 * x1) * (y2 - y3) - (y0 - y1) * (x2 * y3 - y2 * x3)) / denom
    except ZeroDivisionError:
        return
    return [x, y]


#Check if segments have points overlapping in the same direction
def onsegment(p, q, r):
    if (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y)):
        return 1
    return 0

#Check if segments cross each other
def intersection(p, q, r):
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x)*(r.y - q.y)
    if val == 0:
        return 0
    if val > 0:
        return 1
    else:
        return 2
    
#Check if segments intersect (cross or overlap)
def dointersect(p1, q1, p2, q2):
    o1 = intersection(p1, q1, p2)
    o2 = intersection(p1, q1, q2)
    o3 = intersection(p2, q2, p1)
    o4 = intersection(p2, q2, q1)

    if (o1 != o2 and o3 != o4):
        return True
    if (o1 == 0 and onsegment(p1, p2, q1)):
        return True
    if (o2 == 0 and onsegment(p1, q2, q1)):
        return True
    if (o3 == 0 and onsegment(p2, p1, q2)):
        return True
    if (o4 == 0 and onsegment(p2, q1, q2)):
        return True
    return False

#Class for points
class point:
    x = None
    y = None
    def __init__(self,v1,v2):
        self.x = v1
        self.y = v2

    
##########################################################################    
#Save countiguous segments on a list
def generateSegmGroups(lines):
    n = np.shape(lines)[0]
    checked = np.zeros((n), dtype = bool) #list to verify if a segment was already checked, starting with false to all
    segm_groups = [] #list of group of segments
    #lines = connected_lines
    
    def getIntersect(lines, i, path):
        #p1 = point(lines[i][0], lines[i][1])
        #q1 = point(lines[i][2], lines[i][3])
    
        for j in range(0,np.shape(lines)[0]):
            if ((j not in path) and (checked[j] == False)):
                #if checked[j]: return
                
                connected = False
                if (lines[i][0],lines[i][1]) == (lines[j][0],lines[j][1]):
                    connected = True
                elif (lines[i][0],lines[i][1]) == (lines[j][2],lines[j][3]):
                    connected = True
                elif (lines[i][2],lines[i][3]) == (lines[j][0],lines[j][1]):
                    connected = True
                elif (lines[i][2],lines[i][3]) == (lines[j][2],lines[j][3]):
                    connected = True
                    
                if connected:
                    checked[j] = True
                    path.append(j)
                    getIntersect(lines, j, path)
                
                #value = findIntersection(lines[i], lines[j])
                #if value != False:
                #    print(lines[j],lines[i])
                #    print(value)
                
                #p2 = point(lines[j][0], lines[j][1])
                #q2 = point(lines[j][2], lines[j][3])
                '''
                if dointersect(p1,q1,p2,q2):
                    print('connected \n')
                    checked[j] = True
                    path.append(j)
                    getIntersect(lines, j, path) '''
    
    for i in range(0,n):
        if (checked[i] == False):
            checked[i] = True
            path = []
            path.append(i)
            getIntersect(lines, i, path)
            segm_groups.append(path)
            
    return segm_groups
        
    
##########################################################################    
#Get distance and angle of all lines regarding north
def getLineAngles(lines):
    n = np.shape(lines)[0]
    angles = np.zeros((n, 2), np.float64)
    for i in range(0, n):
        length = np.sqrt(np.power((lines[i][2] - lines[i][0]), 2) + np.power((lines[i][3] - lines[i][1]), 2))
        
        '''if (lines[i][1]-lines[i][3]) == 0:
            angle = 90
        elif (lines[i][0]-lines[i][2]) == 0:
            angle = 0
        else:
            '''
        angle = math.atan((lines[i][2]-lines[i][0])/(lines[i][1]-lines[i][3]))/(theta)
        
        if (angle < 0): angle = angle + 360
        if (angle > 180): angle = angle -180
        if math.isnan(angle): angle = 0
        
        angles[i,0] = np.uint8(angle)
        angles[i,1] = length
    return angles

    
    
    
##########################################################################    
#Get regression line of group of segments
def regressionGroupSegments(segm_groups, lines, mode = 'canvas', regression = 'PCA'):
    regression_lines = []
    #lines0 = lines
    #lines = connected_lines
    for i in range(0, np.size(segm_groups)):
        X = []
        Y = []
        if mode == 'vertices':
            X.append(lines[segm_groups[i][0]][0])
            Y.append(lines[segm_groups[i][0]][1])
            for j in segm_groups[i]:
                X.append(lines[j][2])
                Y.append(lines[j][3])
                
            x0 = np.min(X)
            y0 = np.min(Y)
        else:
            #Get canvas extensions
            y= []
            x= []
            for j in segm_groups[i]:
                x.append(lines[j][0])
                x.append(lines[j][2])
                y.append(lines[j][1])
                y.append(lines[j][3])
                
            x0 = np.min(x)
            x1 = np.max(x)
            y0 = np.min(y)
            y1 = np.max(y) 
            canvas = np.zeros((y1-y0+1,x1-x0+1), np.uint8)
            #Draw the segments in the canvas
            for j in segm_groups[i]:
                canvas = cv2.line(canvas, (lines[j][0]-x0,lines[j][1]-y0), (lines[j][2]-x0,lines[j][3]-y0), 1)
            #Get the position of each pixel drawn in the canvas 
            for m in range(0, np.shape(canvas)[0]):
                for n in range(0, np.shape(canvas)[1]):
                    if canvas[m][n] > 0:
                        Y.append(m)
                        X.append(n)
                    
        #least square plane calc
        if regression == 'linear':
            Y = np.reshape(Y, (np.size(Y),1))
            A = np.vstack([X,np.ones(np.size(X))]).T
            m, c = np.linalg.lstsq(A, Y, rcond=None)[0]
            #plt.plot(X, Y*-1, 'bo', label="Data")
            #plt.plot(X, (m*X+c)*-1, 'r--',label="Least Squares")
            #plt.show() 
            line = [X[0] + x0, int(m*X[0]+c)+y0, X[-1]+x0, int(m*X[-1]+c)+y0]
        elif regression == 'PCA':
            pca = PCA(n_components=1)
            data = np.transpose(np.asarray((X,Y)))
            pca.fit(data)
            X_pca = pca.transform(data)
            X_new = pca.inverse_transform(X_pca)
            
            
            #plt.scatter(data[:, 0], -data[:,1])
            #plt.scatter(X_new[:,0], -X_new[:,1])
            #plt.show()
            #plt.plot(X, (m*X+c)*-1, 'r--',label="Least Squares")
            
            #X_new = X_new * [1,-1]
            
            # Obter os extremos horizontais
            p = np.where(X_new[:,0] == np.amax(X_new[:,0]))[0][0]
            q = np.where(X_new[:,0] == np.amin(X_new[:,0]))[0][0]
            if p == q: # Verificar se a linha é vertical para calcular os extremos verticais
                p = np.where(X_new[:,1] == np.amax(X_new[:,1]))[0][0]
                q = np.where(X_new[:,1] == np.amin(X_new[:,1]))[0][0]
            
            line = [int(X_new[p][0] + x0), int(X_new[p][1]+y0), int(X_new[q][0]+x0), int(X_new[q][1]+y0)]
            
        regression_lines.append(line)
        
    return np.reshape(regression_lines, (np.shape(regression_lines)[0], 4))



        

##########################################################################    
#Printbsegments on a canvas      
def drawLines(lines, angles, shape, image = None, mode = 'color'):
    if image.any() == None:
        canvas = np.zeros(shape, np.uint8)
    else:
        if np.size(np.shape(image)) == 2:
            canvas = cv2.cvtColor(np.uint8(image), cv2.COLOR_GRAY2BGR)
        elif np.size(np.shape(image)) == 3:
            canvas = image

    for i in range(0,np.shape(lines)[0]):
        if np.size(shape) == 2:
            color = (255, 255, 255)
        else:
            if mode == 'color':
                color = colorsys.hsv_to_rgb(angles[i][0]/180,1,1)
            else:
                color = (0,0,0)
            color = (color[0]*255, color[1]*255, color[2]*255)
        canvas = cv2.line(canvas, (lines[i][0],lines[i][1]), (lines[i][2],lines[i][3]), (int(color[0]),int(color[1]),int(color[2])), 2)
      
    return canvas
            
##########################################################################    
#Print the group of segments on a canvas, should generate a similar image to the lines without connection      
def drawLineGroups(segm_groups, segm_group_angles, lines, shape, image = None):
    if image.any() == None:
        canvas = np.zeros(shape, np.uint8)
    else:
        if np.size(np.shape(image)) == 2:
            canvas = cv2.cvtColor(np.uint8(image), cv2.COLOR_GRAY2BGR)
        elif np.size(np.shape(image)) == 3:
            canvas = image
      
    for i in range(0, np.size(segm_groups)):
        color = colorsys.hsv_to_rgb(segm_group_angles[i][0]/180,1,1)
        color = (color[0]*255, color[1]*255, color[2]*255)
        for j in segm_groups[i]:
            canvas = cv2.line(canvas, (lines[j][0],lines[j][1]), (lines[j][2],lines[j][3]), color, 2)
    
    return canvas


def ComputeFractureStatistics(lines, threshold, offset):
    data = []
    #offset = 20
    #lines = connected_lines
    check = np.zeros((np.shape(lines)[0]+1), dtype = bool)
    maxH = int(np.shape(threshold)[0]/offset)+1
    maxV = int(np.shape(threshold)[1]/offset)+1
    
    
    def checkArea(x0,x1,y0,y1, lines, k, check):
        count = 0
        vertices = []
        if lines[k][0] >= x0 and lines[k][0] <= x1 and lines[k][1] >= y0 and lines[k][1] <= y1:
            vertices.append(lines[k][0])
            vertices.append(lines[k][1])
            count = count + 1
        if lines[k][2] >= x0 and lines[k][2] <= x1 and lines[k][3] >= y0 and lines[k][3] <= y1:
            vertices.append(lines[k][2])
            vertices.append(lines[k][3])
            count = count + 1
            
        if count == 2:
            check[k] = True
            return np.reshape(vertices, (4,1))
        
        if count == 1:
            border0 = [x0, y0, x1, y0]
            intersection = findIntersection(border0,lines[k])
            if intersection != False: vertices.append(intersection)
                
            border1 = [x1, y0, x1, y1]
            intersection = findIntersection(border1,lines[k])
            if intersection != False: vertices.append(intersection)
                
            border2 = [x0, y1, x1, y1]
            intersection = findIntersection(border2,lines[k])
            if intersection != False: vertices.append(intersection)
                
            border3 = [x0, y1, x0, y0]
            intersection = findIntersection(border3,lines[k])
            if intersection != False: vertices.append(intersection)
            
        if count == 2:
            return np.reshape(vertices, (4,1))
        else:
            False

    for i in range(0,maxV):
        for j in range(0,maxH):
            x0 = i*offset
            x1 = (i+1)*offset
            y0 = j*offset
            y1 = (j+1)*offset
            box = []
            for k in range(0, np.shape(lines)[0]):
                if check[k] == False:
                    line = checkArea(x0,x1,y0,y1, lines, k, check)            
                    if np.size(line) == 4:
                        #print(line)
                        box.append(line)
            
            length = 0
            for row in box:
                length_line = compute_distance(row[0],row[1],row[2],row[3])
                if length_line > 1:
                    length += length_line
                  
            intensity_area = threshold[y0:y1,x0:x1]
            intensity_area = abs(np.sum(intensity_area)/255/np.power(offset,2)-1)
            
            intensity_line = length/np.power(offset,2)
            spacing_area = np.power(offset,2)/(offset + length)
            intensity_line = np.reshape(intensity_line,(1))[0]
            spacing_area = np.reshape(spacing_area, (1))[0]
            data.append(np.reshape((x0,y0,x1,y1,intensity_line,spacing_area, intensity_area),(1, 7)))
    data = np.reshape(data, (np.shape(data)[0],7))
    return data
        

def fractureAreaPlot (data, max_value, canvas, mode = 'raster', index = 'intensity'):
    #max_value = np.max(data[:,4])
    #index = 'spacing'
    #max_value = 20
    if mode == 'raster':
        canvas = np.full((np.shape(canvas)[0], np.shape(canvas)[1], 3), 255, dtype=np.uint8)
    
    for row in data:
        if row[4] == 0.0 and index == 'intensity':
            color = (0,255,178)
        elif row[5] == max_value and index == 'spacing':
            color = (0,102,255)
            
        else:
            if index == 'intensity':
                color = colorsys.hsv_to_rgb((0.45+row[4]/max_value*0.20),1,1)
                color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            elif index == 'spacing':
                color = colorsys.hsv_to_rgb((0.45+row[5]/max_value*0.20),1,1)
                color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            elif index == 'intensity_area':
                color = colorsys.hsv_to_rgb((0.45+row[6]*0.20),1,1)
                color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            else:
                color = (0,0,0)
            
        if mode == 'svg':
            canvas.add(canvas.rect((row[0], row[1]), (20, 20), fill=svgwrite.rgb(int(color[2]), int(color[1]), int(color[0]), 'RGB'))) #stroke=svgwrite.rgb(int(color[0]), int(color[1]), int(color[2]), 'RGB'),
            canvas.add(canvas.text(str(round(row[4],2)), insert=(row[0]+2,row[1]+15), stroke='none', fill=svgwrite.rgb(15, 15, 15, '%'), font_size='9px', font_weight="bold"))
        elif mode == 'raster':
            canvas = cv2.rectangle(canvas, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), color, -1)
    #frac.show_image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    #frac.show_image(canvas)
    return canvas


def boxScale (size, x, y, dwg):  
    dwg.add(dwg.text(str(size)+'m', insert=(2+x,y), stroke='none', fill=svgwrite.rgb(15, 15, 15, '%'), font_size='9px', font_weight="bold"))
    dwg.add(dwg.text(str(size)+'m', insert=(22+x,10+y), stroke='none', fill=svgwrite.rgb(15, 15, 15, '%'), font_size='9px', font_weight="bold"))
    dwg.add(dwg.rect((x, 1+y), (size,size), fill=svgwrite.rgb(255, 255, 255, 'RGB'), stroke=svgwrite.rgb(0, 0, 0, 'RGB')))
    return

def barColor(max_value, x, y, dwg, text):
    color1 = colorsys.hsv_to_rgb((0.45+max_value/max_value*0.15),1,1)
    color1 = svgwrite.rgb(int(color1[2]*255), int(color1[1]*255), int(color1[0]*255),"RGB")
    color2 = colorsys.hsv_to_rgb((0.45+(max_value*0.5)/max_value*0.15),1,1)
    color2 = svgwrite.rgb(int(color2[2]*255), int(color2[1]*255), int(color2[0]*255), "RGB")
    color3 = colorsys.hsv_to_rgb((0.45+0/max_value*0.15),1,1)
    color3 = svgwrite.rgb(int(color3[2]*255), int(color3[1]*255), int(color3[0]*255), "RGB")
    
    vert_grad = svgwrite.gradients.LinearGradient(start=(0, 0), end=(0,1), id="vert_lin_grad")
    vert_grad.add_stop_color(offset='0%', color=color1, opacity=None)
    vert_grad.add_stop_color(offset='50%', color=color2, opacity=None)
    vert_grad.add_stop_color(offset='100%', color=color3, opacity=None)
    dwg.defs.add(vert_grad)
    # draw a box and reference the above gradient definition by #id
    dwg.add(dwg.rect((20+x, y), (10, 100),
        stroke=svgwrite.rgb(10, 10, 16, '%'),
        fill="url(#vert_lin_grad)"
    ))
    
    dwg.add(dwg.text('máximo', insert=(31+x,2+y), stroke='none', fill=svgwrite.rgb(15, 15, 15, '%'), font_size='9px', font_weight="bold"))
    dwg.add(dwg.text('mínimo', insert=(31+x,102+y), stroke='none', fill=svgwrite.rgb(15, 15, 15, '%'), font_size='9px', font_weight="bold"))
    dwg.add(dwg.text(text, insert=(x-10,115+y), stroke='none', fill=svgwrite.rgb(15, 15, 15, '%'), font_size='12px', font_weight="bold"))
    return



def rosechartPlot(angles, title):
    
    bin_edges = np.arange(-5, 366, 10)
    number_of_strikes, bin_edges = np.histogram(angles, bin_edges)
    
    number_of_strikes[0] += number_of_strikes[-1]
    
    half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
    two_halves = np.concatenate([half, half])
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='polar')
    
    ax.bar(np.deg2rad(np.arange(0, 360, 10)), two_halves, 
           width=np.deg2rad(10), bottom=0.0, color='.8', edgecolor='k')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 10), labels=np.arange(0, 360, 10))
    #ax.set_rgrids(np.arange(1, two_halves.max() + 1, 2), angle=0, weight= 'black')
    ax.set_title(title, y=1.10, fontsize=15)
    #plt.show()
    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.show()
    
    return plot

#'''
def loadEdgeData():
    import csv
    import numpy as np
    File = 'Results.csv'
    with open(File) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        table = []
        for row in csv_reader:
            #print(row)
            segment = []
            #lineId = row[1]
            
            if line_count > 0:
                segment.append(int(float(row[1])))
                segment.append(float(row[3]))
                segment.append(float(row[4]))
                segment.append(float(row[8]))
                table.append(segment)
                
            
            line_count += 1
        table = np.reshape(table, (int(np.size(table)/4), 4))
            
        
        #826/2.75
        #898/2.99
        
        table[:,1] = table[:,1]*826/2.75
        table[:,2] = table[:,2]*898/2.99
        table[:,3] = table[:,3]*898/2.99
        
        
    #canvas = np.full((np.shape(image)[0], np.shape(image)[1], 3), 255)
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    max_value = np.max(table[:,3])
    
    import svgwrite
    dwg = svgwrite.Drawing('Ridge.svg', profile='tiny')
    lines = []
    line_aux = []
    iniciolinha = True
    fract_id = 0
    aux = -1
    count = 0
    count2 = 0
    for i in range(np.shape(table)[0]):
        #color = colorsys.hsv_to_rgb((0.45+table[i][3]/max_value*0.15),1,1)
        #color = [int(color[0]*255), int(color[1]*255), int(color[2]*255)]
        #canvas = cv2.circle(canvas, (int(table[i][1]), int(table[i][2])), int(table[i][3]/2), color, -1)
        #print(int(table[i][3]*300))
        aux = table[i][0]
    
    
        if fract_id != aux:
            iniciolinha = True
            count = 2
            fract_id = aux
            dwg.add(dwg.text(str(int(table[i][0])), insert=(table[i][1]+3,table[i][2]+3), stroke='none', fill=svgwrite.rgb(15, 15, 15, '%'), font_size='3px'))
        
        
        
        if i != 0 and iniciolinha == False:
            line_aux.append(table[i-1][1])
            line_aux.append(table[i-1][2])
            line_aux.append(table[i][1])
            line_aux.append(table[i][2])
            lines.append(line_aux)
            line_aux = []
        
        if count >= 1:
            count-= 1
        else:
            iniciolinha = False
            count2+=1
            
            
         
         
    lines = np.reshape(lines, (np.shape(lines)[0],np.shape(lines)[1]))
    
    
    for i in range(np.shape(lines)[0]):
        dwg.add(dwg.line(start=(lines[i][0],lines[i][1]), end=(lines[i][2],lines[i][3]), stroke=svgwrite.rgb(10, 10, 16, '%'), stroke_width=0.5))
    dwg.save()
         
    with open('aperture.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['id', 'x', 'y', 'width'])        
        for i in range(np.shape(table)[0]):
            if i > 0:
                spamwriter.writerow([int(table[i][0]), table[i][1], table[i][2], table[i][3]])
    
    
    canvas = np.uint8(canvas)
    cv2.imwrite('ridge.png', canvas)
    #frac.show_image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
     
#'''



def loadAtlasFile(File,offsetX,offsetY, k, segm_groups = []):
    import csv
    lines = []
    segment = []
    with open(File) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        line_count = 0
        for row in csv_reader:
            if line_count < 3: #Print the content befor the line 4 for header
                print(row)
            else:
                collumn_count = 0
                segment = []
                for collumn in row:
                    if collumn_count > 2: #Ignore collumn indexes before line 3
                        try:
                            value = float(collumn.replace(",",".")) #Get the value in each collumn and replace the commas for dots
                            segment.append(value)
                        except ValueError:
                            print("", end = "")
                    collumn_count = collumn_count + 1
            segment = np.reshape(segment, (int(np.size(segment)/3),3))
            
            segm_group_row = []
            
            #Get individual segments
            for i in range(0,np.shape(segment)[0]-1):
               lines.append([segment[i][0],-segment[i][2],segment[i+1][0],-segment[i+1][2]])
               segm_group_row.append(k)
               k = k + 1

            if np.size(segm_group_row) > 0:
                segm_groups.append(segm_group_row)
                #print(lines)
            line_count = line_count + 1
    
    lines = np.reshape(lines, (np.shape(lines)[0],4))
    
    minX = min(np.append(lines[:,0],lines[:,2]))
    minY = min(np.append(lines[:,1],lines[:,3]))
    maxX = max(np.append(lines[:,0],lines[:,2]))
    maxY= max(np.append(lines[:,1],lines[:,3]))
    
    for i in range(0,np.shape(lines)[0]):
        lines[i][0] = (lines[i][0]-minX+offsetX)
        lines[i][1] = (lines[i][1]-minY+offsetY)
        lines[i][2] = (lines[i][2]-minX+offsetX)
        lines[i][3] = (lines[i][3]-minY+offsetY)
        
    maxY = max(np.append(lines[:,1],lines[:,3]))
    maxX = max(np.append(lines[:,0],lines[:,2]))
        
    return lines, maxX, maxY, segm_groups

'''




#show_image(cv2.cvtColor(cv2.addWeighted(canvas4, 0.4, canvas0, 0.5, 0.0), cv2.COLOR_BGR2RGB))

#cv2.imwrite("intensity3.tif", cv2.addWeighted(canvas0, 0.5, canvas4, 0.5, 0.0))
show_image(cv2.cvtColor(canvas4, cv2.COLOR_BGR2RGB))
cv2.imwrite("intensity.png", canvas4)


cv2.imwrite("intensity.tif",canvas4)

for row in data:
    if row[5] == spacing_max:
        color = (255,255,255)
    else:
        color = colorsys.hsv_to_rgb((0.60-row[5]/spacing_max*0.15),1,1)
        color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
    cv2.rectangle(canvas4, (int(row[0]),int(row[1])),(int(row[2]),int(row[3])), color, -1)
show_image(cv2.cvtColor(cv2.addWeighted(canvas4, 0.4, canvas0, 0.5, 0.0), cv2.COLOR_BGR2RGB))


#cv2.rectangle(canvas4, (x0,y0),(x1,y1),(0,255,255),3)
show_image(cv2.cvtColor(cv2.addWeighted(canvas4, 0.4, canvas0, 0.5, 0.0), cv2.COLOR_BGR2RGB))
cv2.imwrite("intensity.tif",cv2.addWeighted(canvas4, 0.4, cv2.bitwise_not(canvas0), 0.5, 0.0))



##########################################################################    
#Plot wind rose with directions

import scipy.stats as st


N = 13
#np.deg2rad(segm_group_angles)
#hist = np.histogram(segm_group_angles[:,0], bins = 9)

#math.degrees(st.circmean(np.deg2rad(segm_group_angles)))
#math.degrees(st.circvar(np.deg2rad(segm_group_angles)))

bin_width = 10


theta1 = np.linspace(bin_width/2*np.pi/180, np.pi-bin_width/2*np.pi/180, 180/bin_width, endpoint=True)


#for i in range(0, np.shape(regression_lines)[0]):  
#    angle = segm_group_angles[i][0]
#    
#    index = int(angle/180*N)    
#    width[index] = width[index] + angle
#    radii[index] = radii[index] + 1
#    




rosechartPlot(segm_group_angles[:,0], bin_width, "rose.png",'Fraturas - N = ' + str(np.size(segm_group_angles[:,0])))
rosechartPlot(angles_0, bin_width, "rose0.png",'Fraturas até 10m - N = ' + str(np.size(angles_0)))
rosechartPlot(angles_10, bin_width, "rose10.png",'Fraturas entre 10 e 25m - N = ' + str(np.size(angles_10)))
rosechartPlot(angles_25, bin_width, "rose25.png",'Fraturas maiores que 25m - N = ' + str(np.size(angles_25)))

import pycircstat as cs

cs.std(np.dot(angles_0,np.pi/180))*180/np.pi

np.min(segm_group_angles[:,1])

angles_0 = []
for i in range(0,np.size(segm_group_angles[:,0])):
    if segm_group_angles[:,1][i] <= 10:
        #print(segm_group_angles[:,0][i])
        angles_0.append(segm_group_angles[:,0][i])
plt.title('Fraturas até 10m - N = ' + str(np.size(angles_0)))
plt.xlabel("Ângulos (grau)")
plt.ylabel("Frequencia")
plt.grid(axis='y')
plt.hist(angles_0,180)
plt.savefig("hist0.png")


angles_10 = []
for i in range(0,np.size(segm_group_angles[:,0])):
    if segm_group_angles[:,1][i] > 10 and segm_group_angles[:,1][i] <= 25:
        #print(segm_group_angles[:,0][i])
        angles_10.append(segm_group_angles[:,0][i])
plt.title('Fraturas de 10 a 25m - N = ' + str(np.size(angles_10)))
plt.xlabel("Ângulos (grau)")
plt.ylabel("Frequencia")
plt.grid(axis='y')
plt.hist(angles_10,180)
plt.savefig("hist10.png")

angles_25 = []
for i in range(0,np.size(segm_group_angles[:,0])):
    if segm_group_angles[:,1][i] > 25:
        #print(segm_group_angles[:,0][i])
        angles_25.append(segm_group_angles[:,0][i])
plt.title('Fraturas maiores que 25m - N = ' + str(np.size(angles_25)))
plt.xlabel("Ângulos (grau)")
plt.ylabel("Frequencia")
plt.grid(axis='y')
plt.hist(angles_25,180)
plt.savefig("hist25.png")


plt.title('Fraturas - N = ' + str(np.size(segm_group_angles[:,0])))
plt.xlabel("Ângulos (grau)")
plt.ylabel("Frequencia")
plt.grid(axis='y')
plt.hist(segm_group_angles[:,0],100)
plt.savefig("hist.png")


##########################################################################    
#Load lines from Atlas




lines0, maxX, maxY, segm_groups = loadAtlasFile("Interpretacao5_Norte.txt",0,0,0,[])
lines1, maxX, maxY, segm_groups = loadAtlasFile("Interpretacao5_Sul.data",maxX, maxY, np.shape(lines0)[0], segm_groups)
lines = np.concatenate((lines0,lines1), axis=0)

regression_lines = linearRegressionGroupSegments(segm_groups, lines, 'vertices')
segm_group_angles = getLineAngles(regression_lines)




canvas0 = np.full((int((maxY)), int((maxX)),3), 255, np.uint8)
dwg = svgwrite.Drawing('fractures.svg', profile='tiny')
for i in range(0,np.shape(lines)[0]):
    #color = colorsys.hsv_to_rgb(angles[i][0]/180,1,1)
    #color = (color[0]*255, color[1]*255, color[2]*255)
    dwg.add(dwg.line(start=(lines[i][0],lines[i][1]), end=(lines[i][2],lines[i][3]), stroke=svgwrite.rgb(10, 10, 16, '%'), stroke_width=0.5))
    #canvas0 = cv2.line(canvas0, (int(lines[i][0]),int(lines[i][1])), (int(lines[i][2]),int(lines[i][3])),(0,0,0), 1)
    #canvas = cv2.line(canvas, (int((minX+lines[i][0])*10),int((minY+lines[i][1])*10)), (int((minX+lines[i][2])*10),int((minY+lines[i][3])*10)),(0,255,255), 1)
dwg.add(dwg.rect((50,400),(50,10), stroke_width=1, stroke=svgwrite.rgb(10, 10, 16, '%'), fill=svgwrite.rgb(10, 10, 16, '%')))
dwg.add(dwg.rect((100,400),(50,10), stroke_width=1, stroke=svgwrite.rgb(10, 10, 16, '%'), fill=svgwrite.rgb(100, 100, 100, '%')))
dwg.add(dwg.text('0', insert=(50,430), stroke='none', fill=svgwrite.rgb(15, 15, 15, '%'), font_size='9px', font_weight="bold"))
dwg.add(dwg.text('50', insert=(100,430), stroke='none', fill=svgwrite.rgb(15, 15, 15, '%'), font_size='9px', font_weight="bold"))
dwg.add(dwg.text('100', insert=(150,430), stroke='none', fill=svgwrite.rgb(15, 15, 15, '%'), font_size='9px', font_weight="bold"))
dwg.add(dwg.image("NorthArrow_04.svg", (50, 270), size=(60,60)))
dwg.save()
#show_image(canvas4)
show_image(cv2.cvtColor(canvas0, cv2.COLOR_BGR2RGB))

cv2.imwrite("fractures.png",canvas0)
'''