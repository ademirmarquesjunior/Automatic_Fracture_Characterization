# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:08:00 2020

@author: adeju
"""

import numpy as np
import csv
import rasterio as rt
from rasterio.enums import Resampling
import shapefile
# from rasterio.plot import show
# import cv2
# import svgwrite
# import matplotlib.pyplot as plt


def load_image(file, scale):
    dataset = rt.open(file)

    data = dataset.read(
        out_shape=(
            dataset.count,
            int(dataset.height * scale),
            int(dataset.width * scale)
        ),
        resampling=Resampling.bilinear
    )

    transform = dataset.transform * dataset.transform.scale(
        (dataset.width / data.shape[-1]),
        (dataset.height / data.shape[-2])
    )

    image = np.moveaxis(data, 0, -1)
    crs = dataset.profile['crs']
    width = np.shape(image)[1]
    height = np.shape(image)[0]
    count = np.shape(image)[2]

    new_dataset = rt.open("temp/temp.tif", 'w', driver='GTiff',
                          height=height, width=width,
                          count=count, dtype=str(image.dtype),
                          crs=crs,
                          transform=transform)

    return image, new_dataset

# save_image(image,address,dataset.profile['crs'],dataset.profile['transform'])


def save_image(image, file, crs, transform):
    width = np.shape(image)[1]
    height = np.shape(image)[0]

    try:
        count = np.shape(image)[2]
        array = np.moveaxis(image, 2, 0)
    except Exception:
        count = 1
        array = np.reshape(image, (1, np.shape(image)[0],
                                   np.shape(image)[1]))

    new_dataset = rt.open(file, 'w', driver='GTiff',
                          height=height, width=width,
                          count=count, dtype=str(array.dtype),
                          crs=crs,
                          transform=transform)

    new_dataset.write(array)
    new_dataset.close()

    return


def load_atlas_file(File, offsetX, offsetY, k=0, segm_groups=[]):
    '''
    Load Mosis XP project file.

    Parameters
    ----------
    File : str
        File address of Mosis XP project file.
    offsetX : float64
        Offset for the X coordinates.
    offsetY : float64
        Offset for the X coordinates.
    k : int64, optional
        Start index for segment_groups array. The default is 0.
    segm_groups : Array of lists, optional
        Group of segments previously used. The default is [].

    Returns
    -------
    lines : Array of float64
        Array of individual segments. [[x0, y0, x1, y1], ...].
    maxX : TYPE
        X extension limit.
    maxY : TYPE
        Y extension limit.
    segm_groups : TYPE
        Group of segments.

    '''
    lines = []
    segment = []

    with open(File) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        line_count = 0
        for row in csv_reader:
            if line_count < 7:  # Print the content befor the line 4 for header
                print(row)
            else:
                collumn_count = 0
                segment = []
                for collumn in row:
                    # Ignore collumn indexes before line 3
                    if collumn_count > 2:
                        try:
                            # Get the value in each collumn and replace the
                            # commas for dots
                            value = float(collumn.replace(",", "."))
                            segment.append(value)
                        except ValueError:
                            print("", end="")
                    collumn_count = collumn_count + 1
            segment = np.reshape(segment, (int(np.size(segment)/3), 3))

            segm_group_row = []

            # Get individual segments
            for i in range(0, np.shape(segment)[0]-1):
                lines.append([segment[i][0], -segment[i][2], segment[i+1][0],
                             -segment[i+1][2]])
                segm_group_row.append(k)
                k = k + 1

            if np.size(segm_group_row) > 0:
                segm_groups.append(segm_group_row)
                # print(lines)
            line_count = line_count + 1

    lines = np.reshape(lines, (np.shape(lines)[0], 4))

    minX = min(np.append(lines[:, 0], lines[:, 2]))
    minY = min(np.append(lines[:, 1], lines[:, 3]))
    maxX = max(np.append(lines[:, 0], lines[:, 2]))
    maxY = max(np.append(lines[:, 1], lines[:, 3]))

    for i in range(0, np.shape(lines)[0]):
        lines[i][0] = (lines[i][0]-minX+offsetX)
        lines[i][1] = (lines[i][1]-minY+offsetY)
        lines[i][2] = (lines[i][2]-minX+offsetX)
        lines[i][3] = (lines[i][3]-minY+offsetY)

    maxY = max(np.append(lines[:, 1], lines[:, 3]))
    maxX = max(np.append(lines[:, 0], lines[:, 2]))

    return lines, maxX, maxY, segm_groups


def load_ridge_data(results_file):
    '''
    Load result files from ImageJ Ridge Detection plugin

    Parameters
    ----------
    results_file : str
        File address for ridge data results. Usually "Results.csv".

    Returns
    -------
    lines : Array of float64
        Array of individual segments. [[x0, y0, x1, y1], ...]
    aperture_data : Array of float64
        Array of aperture data. [[id, x, y, aperture], ...]

    '''
    # results_file = 'Results.csv'
    with open(results_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        aperture_data = []
        for row in csv_reader:
            # print(row)
            segment = []
            # lineId = row[1]

            if line_count > 0:
                segment.append(int(float(row[1])))
                segment.append(float(row[3]))
                segment.append(float(row[4]))
                segment.append(float(row[8]))
                aperture_data.append(segment)

            line_count += 1
        aperture_data = np.reshape(aperture_data,
                                   (int(np.size(aperture_data)/4), 4))

    lines = []
    line_row = []
    line_start = True
    fract_id = 0
    aux = -1
    count = 0
    for i in range(np.shape(aperture_data)[0]):
        aux = aperture_data[i, 0]

        if fract_id != aux:
            line_start = True
            count = 2
            fract_id = aux

        if i != 0 and line_start is False:
            line_row.append(aperture_data[i-1, 1])
            line_row.append(aperture_data[i-1, 2])
            line_row.append(aperture_data[i, 1])
            line_row.append(aperture_data[i, 2])
            lines.append(line_row)
            line_row = []

        if count >= 1:
            count -= 1
        else:
            line_start = False

    lines = np.reshape(lines, (np.shape(lines)[0], np.shape(lines)[1]))

    return lines, aperture_data


def save_shapefile(file_address, regression_lines, segm_group_angles, dataset):
    '''
    Save line segments to shapefiles.

    Parameters
    ----------
    file_address : String
        DESCRIPTION.
    regression_lines : Array of int64
        Array of line segments.
    segm_group_angles : Array of int64
        Array of angles and line lengths.
    dataset : Object
        Rasterio object.

    Returns
    -------
    None.

    '''
    w = shapefile.Writer(file_address, shapeType=3)
    w.field('fracture_i', 'N')
    w.field('fractdir', 'N')
    w.field('fractlength', 'N')

    for i in range(0, np.shape(regression_lines)[0]):
        point0 = dataset.xy(regression_lines[i, 1], regression_lines[i, 0])
        point1 = dataset.xy(regression_lines[i, 3], regression_lines[i, 2])
        # Add record
        w.record(i, segm_group_angles[i, 0], 0)
        # Add geometry
        w.line([[[point0[0], point0[1]], [point1[0], point1[1]]]])

    w.close()
    return
