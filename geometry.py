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

        if lines[i][1] == lines[i][3]:
            angle = 90
        else:
            angle = (math.atan((lines[i][2]-lines[i][0])
                               / (lines[i][1]-lines[i][3])) / (THETA))

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


# Helper function for line intersection on function find_intersection**********

def on_segment(p, q, r):
    # Check if segments have points overlapping in the same direction
    if (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y)):
        return 1
    return 0


def intersection(p, q, r):
    # Check if segments cross each other
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x)*(r.y - q.y)
    if val == 0:
        return 0
    if val > 0:
        return 1
    else:
        return 2


def do_intersect(p1, q1, p2, q2):
    # Check if segments intersect (cross or overlap)
    o1 = intersection(p1, q1, p2)
    o2 = intersection(p1, q1, q2)
    o3 = intersection(p2, q2, p1)
    o4 = intersection(p2, q2, q1)

    if (o1 != o2 and o3 != o4):
        return True
    if (o1 == 0 and on_segment(p1, p2, q1)):
        return True
    if (o2 == 0 and on_segment(p1, q2, q1)):
        return True
    if (o3 == 0 and on_segment(p2, p1, q2)):
        return True
    if (o4 == 0 and on_segment(p2, q1, q2)):
        return True
    return False


class Point:
    # Class for points
    x = None
    y = None

    def __init__(self, v1, v2):
        self.x = v1
        self.y = v2


def find_intersection(line1, line2):
    '''
    Receives two line segments and returns the crossing point if exists

    Parameters
    ----------
    line1 : list
        Two xy coordinates in a list of four itens.
    line2 : TYPE
        Two xy coordinates in a list of four itens.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    x0, y0, x1, y1 = line1
    x2, y2, x3, y3 = line2

    if do_intersect(
            Point(x0, y0), Point(x1, y1), Point(x2, y2),
            Point(x3, y3)) == False:
        return False

    try:
        denom = float((x0 - x1) * (y2 - y3) - (y0 - y1) * (x2 - x3))
        x = ((x0 * y1 - y0 * x1) * (x2 - x3) -
             (x0 - x1) * (x2 * y3 - y2 * x3)) / denom
        y = ((x0 * y1 - y0 * x1) * (y2 - y3) -
             (y0 - y1) * (x2 * y3 - y2 * x3)) / denom
    except ZeroDivisionError:
        return False
    return [x, y]

# 3d geometry
