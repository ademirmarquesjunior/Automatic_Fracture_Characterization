# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:42:03 2019

@author: ADEJUNIOR
"""


import math
import numpy as np

THETA = math.pi/180


def compute_distance(x0, y0, x1, y1):
    '''
    Compute the distance between two point.

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


def get_line_angles(lines):
    '''
    Get distance and angle of all lines regarding north

    Parameters
    ----------
    lines : Array of int64
        Array with structure [[Px Py Qx Qy] ... [Px Py Qx Qy]].

    Returns
    -------
    angles : Array of float64
        Array with angle and length for each line.

    '''
    # Get distance and angle of all lines regarding north
    n = np.shape(lines)[0]
    angles = np.zeros((n, 2), np.float64)
    for i in range(0, n):
        length = compute_distance(lines[i][0], lines[i][1], lines[i][2],
                                  lines[i][3])
        angle = (math.atan((lines[i][2]-lines[i][0])/(lines[i][1]-lines[i][3]))
                 / (THETA))

        if (angle < 0):
            angle = angle + 360
        if (angle > 180):
            angle = angle - 180
        if math.isnan(angle):
            angle = 0

        angles[i, 0] = np.uint8(angle)
        angles[i, 1] = length
    return angles


def compare_angles(base, line1, line2):
    '''
    Measure the angle between two lines or segments given a reference vertex.
    Extends the closest point on line2 to the base point on line1 to measure
    the angle.

    Parameters
    ----------
    base : list
        Vertex on line1 used as reference to obtain the angle.
    line1 : list
        First line or segment.
    line2 : list
        Second line or segment.

    Returns
    -------
    angle : float
        Angle in degrees.
    distance : float
        Distance to the closest point on line2 to the base point.
    x : int
        X position of the closest point on line2.
    y : int
        Y position of the closest point on line2.

    '''
    x0, y0 = base
    if [line1[0], line1[1]] == [x0, y0]:
        px, py = [line1[2], line1[3]]
    else:
        px, py = [line1[0], line1[1]]

    dist0 = compute_distance(x0, y0, line2[0], line2[1])
    dist1 = compute_distance(x0, y0, line2[2], line2[3])

    if dist0 < dist1:
        px = px - x0
        py = py - y0
        qx, qy = [float(line2[2]) - x0, float(line2[3]) - y0]
        aux = ((px*qx) + (py*qy))/(math.sqrt(
            math.pow(px, 2) + math.pow(py, 2))*math.sqrt(math.pow(qx, 2) +
                                                         math.pow(qy, 2)))
        try:
            angle = math.acos(aux)*180/math.pi
        except Exception:
            if (aux) > 0:
                angle = 0
            if (aux) < 0:
                angle = 180
        return (angle, dist0, line2[0], line2[1])
    else:
        px = px - x0
        py = py - y0
        qx, qy = [line2[0] - x0, line2[1] - y0]
        aux = ((px*qx) + (py*qy))/(math.sqrt(
            math.pow(px, 2) + math.pow(py, 2))*math.sqrt(math.pow(qx, 2) +
                                                         math.pow(qy, 2)))
        try:
            angle = math.acos(aux)*180/math.pi
        except Exception:
            if (aux) > 0:
                angle = 0
            if (aux) < 0:
                angle = 180
        return (angle, dist1, line2[2], line2[3])