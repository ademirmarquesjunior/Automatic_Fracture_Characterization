# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:50:19 2019

@author: adejunior
"""
import cv2
import numpy as np
import PySimpleGUI as sg
import fracture_detection_hough as frac

import matplotlib.pyplot as plt
import math
# import pandas as pd

import base64
from io import BytesIO
import timeit

import rasterio
import utm

# from pathlib import Path, PureWindowsPath

# gray2 = gray[0:500,0:500]
# cv2.imwrite("temp.png", gray[0:500,0:500])


menu_def = [['File', ['Open', 'Save', 'Exit']],
            ['Help', ['About...', 'Reload']], ]

load_buttons = [sg.Button('Original', key="_Bt_original_", disabled=True,
                          button_color=('black', 'springgreen4')),
                sg.Button('Blur', key="_Bt_smoothed_", disabled=True,
                          button_color=('black', 'springgreen4')),
                sg.Button('Threshold', key="_Bt_thresholded_", disabled=True,
                          button_color=('black', 'springgreen4')),
                sg.Button('1-pixel', key="_Bt_thinned_", disabled=True,
                          button_color=('black', 'springgreen4')),
                sg.Button('Hough', key="_Bt_lined_", disabled=True,
                          button_color=('black', 'springgreen4')),
                sg.Button('Connected', key="_Bt_connected_", disabled=True,
                          button_color=('black', 'springgreen4')),
                sg.Button('Groups', key="_Bt_groups_", disabled=True,
                          button_color=('black', 'springgreen4')),
                sg.Button('Plot', key="_Bt_rosechart_plot_", disabled=True,
                          button_color=('black', 'springgreen4')),
                sg.Button('Intensity', key="_Bt_intensity_plot_",
                          disabled=True, button_color=('black',
                                                       'springgreen4')),
                sg.Button('Spacing', key="_Bt_spacing_plot_", disabled=True,
                          button_color=('black', 'springgreen4'))
                ]

main_canvas = sg.Graph(canvas_size=(500, 500), graph_bottom_left=(0, 0),
                       enable_events=True, drag_submits=False,
                       graph_top_right=(500, 500), key="_Canvas1_")

vertical_slider = sg.Slider(
                        range=(100, 0), orientation='v', size=(20, 10),
                        enable_events=True, disable_number_display=True,
                        default_value=0, key="_Sl_vertical_")

horizontal_slider = sg.Slider(
                        range=(0, 100), orientation='h', size=(40, 10),
                        enable_events=True, disable_number_display=True,
                        default_value=0, key="_Sl_horizontal_")


layout = [[sg.Menu(menu_def, tearoff=True)],
          [main_canvas, vertical_slider,
           sg.Frame("", [
                [sg.Frame("Image processing", [
                    [sg.Text('Filter size'),
                     sg.Radio('3', "RADIO1"),
                     sg.Radio('5', "RADIO1"),
                     sg.Radio('7', "RADIO1", default=True),
                     sg.Radio('9', "RADIO1"),
                     sg.Radio('11', "RADIO1"),
                     sg.Radio('13', "RADIO1")],
                    [sg.Text('Filter mode'),
                     sg.Radio('Mean', "RADIO2"),
                     sg.Radio('Median', "RADIO2", default=True),
                     sg.Radio('Gaussian', "RADIO2"),
                     sg.Radio('Bilateral', "RADIO2"),
                     ],
                    [sg.Text('Adaptative thresholding'),
                     sg.Radio('Sauvola', "RADIO3", default=True),
                     sg.Radio('Niblack', "RADIO3"),
                     sg.Radio('Phansalkar', "RADIO3", disabled=True),
                     ],
                    [sg.Text('Skeletonization'),
                     sg.Radio('Lee', "RADIO4", default=True),
                     sg.Radio('Zhang', "RADIO4", disabled=True),
                     sg.Radio
                     ],
                    [sg.Button('Image processing', key='_Bt_processing_',
                               disabled=True,
                               size=(50, 1)), ]])],

                # [
                #     sg.Frame("Edge", [[sg.Button('Canny', key="_Bt_Canny_",
                #                                  disabled=True)]]), ],

                [sg.Frame("Hough transform", [
                    [sg.Text('Threshold'), sg.Slider(range=(1, 100),
                                                     orientation='h',
                                                     size=(10, 10),
                                                     default_value=33,
                                                     key="_Sl_threshold_"),
                     sg.Button('Hough', key='_Bt_hough_', disabled=True)],
                    [sg.Text('Min'),
                     sg.Slider(range=(2, 50), orientation='h', size=(10, 10),
                               default_value=2, key="_Sl_min_"),
                     sg.Text('Max'),
                     sg.Slider(range=(2, 50), orientation='h', size=(10, 10),
                               default_value=2, key="_Sl_max_")],
                         ]),
                 ],
                [sg.Frame("Connecting lines", [
                 [sg.Text('radius'),
                  sg.Slider(range=(1, 100), orientation='h', size=(10, 10),
                            default_value=50, key="_Sl_radius_"),
                  sg.InputCombo(('alpha', 'beta', 'distance', 'deviation'),
                                key='_Cb_mode_'),
                  sg.Button('Connect', key='_Bt_connect_', disabled=True)],
                 [sg.Text('alpha'), sg.Slider(range=(0, 180), orientation='h',
                                              size=(10, 10), default_value=120,
                                              key="_Sl_alpha_"),
                  sg.Text('beta'), sg.Slider(range=(0, 180), orientation='h',
                                             size=(10, 10), default_value=90,
                                             key="_Sl_beta_")],
                 ])],
                [sg.Frame("Directional statistics", [
                 [sg.Text('Method'),
                  sg.InputCombo(('points', 'vertices'), key='_Cb_method_'),
                  sg.Text('Regression'), sg.InputCombo(('PCA', 'linear'),
                                                       key='_Cb_regression_'),
                  sg.Button('Rosechart', key='_Bt_rosechart_', disabled=True)]
                 ])],
                [sg.Frame("Fracture statistics - intensity/spacing", [
                 [sg.Text('Box size'), sg.Input(key='_In_size_', size=(5, 5)),
                  sg.InputCombo(('pixels', 'meters'), key='_Cb_measure_'),
                  sg.InputCombo(('raster', 'svg'), key='_Cb_output_'),
                  sg.Button('Process', key='_Bt_fracture_statistics_',
                            disabled=True)]
                 ])],
                # [sg.Frame("Fracture statistics", [
                # [sg.Input(key='_In_point0_', size=(20,10)),
                # sg.Input(key='_In_point1_', size=(20,10)),
                # sg.Button('Scanline', key='_Bt_scanline_', disabled=True)],
                # [sg.Multiline(default_text="", key="_Mt_positions_"),
                # sg.Button('Draw points', key='_Bt_positions_')]
                # ])],

                ])
           ],
          [horizontal_slider],
          load_buttons,
          [sg.Text('', key="_Tx_position_", auto_size_text=True)]
          ]

window = sg.Window('Fraturas', layout)
canvas1 = window.Element("_Canvas1_")
temp = np.zeros(canvas1.CanvasSize)


def updateCanvas(image, hor, ver):
    if np.size(np.shape(image)) == 2:
        image = cv2.cvtColor(np.uint8(image), cv2.COLOR_GRAY2BGR)
    size = canvas1.CanvasSize
    positionX = int(np.shape(image)[1]/100*hor)
    positionY = int(np.shape(image)[0]/100*ver)
    if positionX > np.shape(image)[1] - size[1]:
        positionX = np.shape(image)[1]-size[1]-1
    if positionY > (np.shape(image)[0]-size[0]):
        positionY = np.shape(image)[0]-size[0]-1
    try:
        buffered = BytesIO()
        frac.Image.fromarray(np.uint8(
            image[positionY:size[0]+positionY,
                  positionX:size[1]+positionX])).save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue())
        canvas1.DrawImage(data=encoded, location=(0, 500))
    except Exception as e:
        print(e)
    return image


def mapToImagePosition(map_position):
    if np.size(map_position) == 3:  # UTM coordinates
        try:
            lat, lon = utm.to_latlon(map_position[0], map_position[1],
                                     map_position[2], northern=False)
            image_position = dataset.index(lon, lat)
        except Exception as e:
            print(e)
            return False
    elif np.size(map_position) == 2:  # WGS84 coordinates
        image_position = dataset.index(map_position[1], map_position[0])
    else:
        return False
    return image_position


def destinationPoint(lat1, lon1, distance, bearing, radius=6371000):

    angDist = distance / radius  # angular distance in radians
    bearing = math.radians(bearing)

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2sin = math.sin(lat1) * math.cos(angDist) + math.cos(lat1) * math.sin(
        angDist) * math.cos(bearing)
    lat2 = math.asin(lat2sin)
    y = math.sin(bearing) * math.sin(angDist) * math.cos(lat1)
    x = math.cos(angDist) - math.sin(lat1) * lat2sin
    lon2 = lon1 + math.atan2(y, x)

    lat = math.degrees(lat2)
    lon = math.degrees(lon2)

    return (lat, lon)


while True:
    event, values = window.Read()
    if event is None or event == 'Exit':
        break
    elif event == 'Open':
        hor = int(values["_Sl_horizontal_"])
        ver = int(values["_Sl_vertical_"])

        address = sg.PopupGetFile('Document to open')
        try:
            file = open(address, 'rb').read()
            dataset = rasterio.open(address)
            # sg.Popup(str(dataset.crs + dataset.bounds))
            try:
                image = cv2.imdecode(np.frombuffer(file, np.uint8), 0)
                # temp = image
                temp = updateCanvas(image, hor, ver)
                window.Element("_Bt_processing_").Update(disabled=False)
                window.Element("_Bt_original_").Update(disabled=False)
                window.Element("_Tx_position_").Update(value=str(
                    dataset.crs)+" "+str(dataset.bounds.left,
                                         dataset.bounds.top))
                (dataset.bounds.left, dataset.bounds.top)
                # window.Element("_Bt_scanline_").Update(disabled=False)
            except ValueError:
                sg.Popup(ValueError)
        except Exception as e:
            sg.Popup('Invalid file address. '+str(e))
    elif event == '_Sl_horizontal_' or event == '_Sl_vertical_':
        hor = int(values["_Sl_horizontal_"])
        ver = int(values["_Sl_vertical_"])
        temp = updateCanvas(temp, hor, ver)
    elif event == '_Canvas1_':
        position = values['_Canvas1_']
        hor = int(values["_Sl_horizontal_"])
        ver = int(values["_Sl_vertical_"])
        try:
            size = canvas1.CanvasSize
            positionX = int(np.shape(image)[1]/100*hor)
            positionY = int(np.shape(image)[0]/100*ver)
            if positionX > np.shape(image)[1]-size[1]:
                positionX = np.shape(image)[1]-size[1]-1
            if positionY > (np.shape(image)[0]-size[0]):
                positionY = np.shape(image)[0]-size[0]-1
            x = positionX+position[0]
            y = positionY+abs(position[1]-size[1])
            window.Element("_Tx_position_").Update(value=str(dataset.crs)
                                                   + str(dataset.xy(x, y)))
        except Exception as e:
            print(e)
            window.Element("_Tx_position_").Update(value=str((position[0],
                                                              position[1])))

    elif event == '_Bt_processing_':
        hor = int(values["_Sl_horizontal_"])
        ver = int(values["_Sl_vertical_"])
        sigma = 0
        kernel = 3
        filter_method = ''

        if values[1] is True:
            kernel = 3
        if values[2] is True:
            kernel = 5
        if values[3] is True:
            kernel = 7
        if values[4] is True:
            kernel = 9
        if values[5] is True:
            kernel = 11
        if values[6] is True:
            kernel = 13

        if values[7] is True:
            smooth = cv2.blur(image, (kernel, kernel))
        if values[8] is True:
            smooth = cv2.medianBlur(image, kernel)
        if values[9] is True:
            smooth = cv2.GaussianBlur(image, (kernel, kernel), sigma)
        if values[10] is True:
            smooth = cv2.bilateralFilter(image, kernel, 75, 75)

        if values[11] is True:
            threshold = frac.adaptative_thresholding(smooth, 31, 'sauvola')
        if values[12] is True:
            threshold = frac.adaptative_thresholding(smooth, 31, 'niblack')
        if values[13] is True:
            threshold = frac.adaptative_thresholding(smooth, 31, 'phansalkar')

        if values[14] is True:
            onepixel = frac.skeletonize_image(threshold, True, 'lee')
        if values[15] is True:
            onepixel = frac.skeletonize_image(threshold, True, 'lee')

        temp = updateCanvas(cv2.bitwise_not(onepixel), hor, ver)

        cv2.imwrite("smooth.png", smooth)
        cv2.imwrite("threshold.png", threshold)
        cv2.imwrite("skeleton.png", cv2.bitwise_not(onepixel))

        window.Element("_Bt_smoothed_").Update(disabled=False)
        window.Element("_Bt_thresholded_").Update(disabled=False)
        window.Element("_Bt_hough_").Update(disabled=False)
        window.Element("_Bt_thinned_").Update(disabled=False)
        sg.Popup('Image processing finished')

    elif event == '_Bt_hough_':
        hor = int(values["_Sl_horizontal_"])
        ver = int(values["_Sl_vertical_"])
        theta = math.pi/180
        lines = cv2.HoughLinesP(np.asarray(frac.Image.fromarray(onepixel),
                                           dtype=np.uint8), 1, math.pi/180/2,
                                threshold=int(values["_Sl_threshold_"]),
                                minLineLength=int(values["_Sl_min_"]),
                                maxLineGap=int(values["_Sl_max_"]))
        lines = np.reshape(lines, (np.shape(lines)[0], 4))
        angles = frac.getLineAngles(lines)
        houghlines = frac.drawLines(lines, angles, (np.shape(image)[0],
                                                    np.shape(image)[1], 3),
                                    smooth)
        # temp = houghlines
        temp = updateCanvas(houghlines, hor, ver)
        cv2.imwrite("hough.png", houghlines)
        window.Element("_Bt_lined_").Update(disabled=False)
        window.Element("_Bt_connect_").Update(disabled=False)

    elif event == '_Bt_connect_':
        hor = int(values["_Sl_horizontal_"])
        ver = int(values["_Sl_vertical_"])
        # start = timeit.timeit()
        connected_lines = frac.connectLines(smooth, lines, angles, int(
            values['_Sl_radius_']), float(values['_Sl_alpha_']), float(
                values['_Sl_beta_']), values['_Cb_mode_'])
        # import fracture_detection_hough as frac
        # connected_lines = frac.connectLines(
        #   smooth, lines, angles, 50, 135, 135, 'distance')
        # end = timeit.timeit()
        # print((end - start)*100)
        angles2 = frac.getLineAngles(connected_lines)
        connectionlines = frac.drawLines(np.uint64(connected_lines), angles2,
                                         (np.shape(image)[0], np.shape(
                                             image)[1], 3), image=smooth)

        # frac.show_image(connectionlines)
        # frac.show_image(houghlines)
        # frac.show_image(cv2.addWeighted(connectionlines, 0.4, cv2.cvtColor(
        #   smooth, cv2.COLOR_GRAY2BGR), 0.5, 0.0))

        cv2.imwrite("connected3.png", connectionlines)
        # connectionlines = cv2.addWeighted(connectionlines, 0.4,
        #                   cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR), 0.5, 0.0)

        temp = updateCanvas(connectionlines, hor, ver)
        # cv2.imwrite("connected.png", connectionlines)
        sg.Popup('Connections finished')
        window.Element("_Bt_connected_").Update(disabled=False)
        window.Element("_Bt_rosechart_").Update(disabled=False)
        window.Element("_Bt_fracture_statistics_").Update(disabled=False)

    elif event == '_Bt_rosechart_':
        hor = int(values["_Sl_horizontal_"])
        ver = int(values["_Sl_vertical_"])

        start = timeit.timeit()
        segm_groups = frac.generateSegmGroups(connected_lines)
        regression_lines = frac.regressionGroupSegments(segm_groups,
                                                        connected_lines,
                                                        values['_Cb_method_'],
                                                        values['_Cb_regression_'])
        segm_group_angles = frac.getLineAngles(regression_lines)
        end = timeit.timeit()
        print(end - start)

        pca_lines = frac.drawLines(regression_lines, segm_group_angles, (
            np.shape(image)[0], np.shape(image)[1], 3), image)
        cv2.imwrite('pca_lines.png', pca_lines)

        # frac.show_image()
        # plt.hist(segm_group_angles[:,1])

        segmgroups = frac.drawLineGroups(segm_groups, segm_group_angles,
                                         connected_lines, (np.shape(image)[0],
                                                           np.shape(
                                                               image)[1], 3),
                                         smooth)

        cv2.imwrite('groups.png', segmgroups)

        # frac.show_image(segmgroups)

        window.Element("_Sl_horizontal_").Update(value=0)
        window.Element("_Sl_vertical_").Update(value=0)
        plot = frac.rosechartPlot(segm_group_angles[:, 0], 'Rosechart')
        cv2.imwrite('plot.png', plot)
        plot = cv2.resize(plot, (500, 500))
        temp = updateCanvas(plot, 0, 0)
        window.Element("_Bt_groups_").Update(disabled=False)
        window.Element("_Bt_rosechart_plot_").Update(disabled=False)

    elif event == '_Bt_fracture_statistics_':
        hor = int(values["_Sl_horizontal_"])
        ver = int(values["_Sl_vertical_"])

        try:
            boxsize = float(values['_In_size_'])
        except Exception as e:
            print(e)
            sg.Popup('Insert a valid number!')
            break

        dataset = rasterio.open(address)
        print(dataset.crs)

        if str(dataset.crs) != 'None':
            position_utm = np.asarray(utm.from_latlon(dataset.bounds.left,
                                                      dataset.bounds.top))
            position = utm.to_latlon(float(position_utm[0]),
                                     float(position_utm[1]) + boxsize,
                                     int(position_utm[2]), position_utm[3])
            row, col = dataset.index(position[0], position[1])

            '''
            dataset = rasterio.open('lajedo4.tif')

            inicio = utm.from_latlon(dataset.xy(0,0)[0], dataset.xy(0,0)[1])
            fim = utm.from_latlon(dataset.xy(0,60)[0], dataset.xy(0,60)[1])

            print(frac.compute_distance(inicio[0], inicio[1] , fim[0], fim[1]))
            '''

            boxsize = int(frac.compute_distance(0, 0, row, col))

        # boxsize = 60
        data = frac.ComputeFractureStatistics(connected_lines, threshold,
                                              boxsize)

        intensity_max = np.max(data[:, 4])

        if values['_Cb_output_'] == 'svg':
            import svgwrite
            dwg = svgwrite.Drawing('intensity.svg', profile='tiny')
            intensity = frac.fractureAreaPlot(data, intensity_max, dwg, 'svg',
                                              'intensity')
            frac.barColor(intensity_max, 40, 350, dwg, "Intensity")
            frac.boxScale(20, 50, 500, dwg)
            dwg.save()

            dwg = svgwrite.Drawing('spacing.svg', profile='tiny')
            spacing = frac.fractureAreaPlot(data, boxsize, dwg, 'svg',
                                            'spacing')
            frac.barColor(intensity_max, 40, 350, dwg, "Spacing")
            frac.boxScale(20, 50, 500, dwg)
            dwg.save()

            temp = updateCanvas(intensity, hor, ver)

        else:
            intensity = frac.fractureAreaPlot(data, intensity_max, image,
                                              'raster', 'intensity')
            intensity = frac.drawLines(connected_lines, angles2, image,
                                       image=intensity, mode='black')
            cv2.imwrite('intensity.png', intensity)
            intensity = cv2.cvtColor(intensity, cv2.COLOR_BGR2RGB)
            # frac.show_image(cv2.cvtColor(intensity, cv2.COLOR_BGR2RGB))

            spacing = frac.fractureAreaPlot(data, boxsize, image, 'raster',
                                            'spacing')
            spacing = frac.drawLines(connected_lines, angles2, image,
                                     image=spacing, mode='black')

            cv2.imwrite('spacing.png', spacing)
            spacing = cv2.cvtColor(spacing, cv2.COLOR_BGR2RGB)
            # frac.show_image(cv2.cvtColor(spacing, cv2.COLOR_BGR2RGB))

            intensity_area = frac.fractureAreaPlot(data, boxsize, image,
                                                   'raster', 'intensity_area')

            for i in range(0, np.shape(image)[0]-1):
                for j in range(0, np.shape(image)[1]-1):
                    if threshold[i, j] == 0:
                        intensity_area[i, j] = [0, 0, 0]

            # frac.show_image(cv2.cvtColor(intensity_area, cv2.COLOR_BGR2RGB))
            cv2.imwrite('intensity_area.png', intensity_area)

            temp = updateCanvas(intensity, hor, ver)

        window.Element("_Bt_intensity_plot_").Update(disabled=False)
        window.Element("_Bt_spacing_plot_").Update(disabled=False)

    elif event == '_Bt_scanline_':
        hor = int(values["_Sl_horizontal_"])
        ver = int(values["_Sl_vertical_"])
        try:
            p0 = np.asarray(values['_In_point0_'].split(), dtype=np.float64)
            p1 = np.asarray(values['_In_point1_'].split(), dtype=np.float64)

            x0, y0 = mapToImagePosition(p0)
            x1, y1 = mapToImagePosition(p1)
            # temp = cv2.line(temp, (y0, x0), (y1, x1), [255,255,255], 10)
            pixels = np.asarray(frac.bresenham_march(threshold, (y0, x0),
                                                     (y1, x1)))
            # plt.figure(figsize=(100,20))
            # plt.plot(pixels1[:,1])
            # plt.savefig("plot.png")

            # temp = cv2.circle(temp, (y0, x0), 30, [255,255,255], 10)
            # frac.show_image(temp)
            temp = updateCanvas(temp, hor, ver)
        except Exception as e:
            print(e)
            sg.Popup("Invalid coordinates")
    elif event == '_Bt_positions_':
        positions = np.asarray(values['_Mt_positions_'].split(),
                               dtype=np.float64)
        positions = np.reshape(positions, (int(np.size(positions)/3), 3))

        for i in range(0, np.shape(positions)[0]-1):
            y, x = mapToImagePosition(positions[i])
            temp = cv2.drawMarker(temp, (x, y), (0, 0, 255),
                                  markerType=cv2.MARKER_TILTED_CROSS,
                                  thickness=1)

        new_dataset = rasterio.open('test1.tif', 'w', driver='GTiff',
                                    height=temp.shape[0], width=temp.shape[1],
                                    count=1, dtype=str(temp.dtype),
                                    crs=dataset.crs,
                                    transform=dataset.transform)
        new_dataset.write(cv2.cvtColor(np.uint8(temp), cv2.COLOR_BGR2GRAY), 1)
        new_dataset.close()

    elif event == 'Reload':
        import fracture_detection_hough as frac

    # Update canvas according to the button
    elif event == '_Bt_original_':
        temp = updateCanvas(image, hor, ver)
    elif event == '_Bt_smoothed_':
        temp = updateCanvas(smooth, hor, ver)
    elif event == '_Bt_thresholded_':
        temp = updateCanvas(threshold, hor, ver)
    elif event == '_Bt_thinned_':
        temp = updateCanvas(cv2.bitwise_not(onepixel), hor, ver)
    elif event == '_Bt_lined_':
        temp = updateCanvas(houghlines, hor, ver)
    elif event == '_Bt_connected_':
        temp = updateCanvas(connectionlines, hor, ver)
    elif event == '_Bt_groups_':
        temp = updateCanvas(segmgroups, hor, ver)
    elif event == '_Bt_rosechart_plot_':
        temp = updateCanvas(plot, 0, 0)
    elif event == '_Bt_intensity_plot_':
        temp = updateCanvas(intensity, hor, ver)
    elif event == '_Bt_spacing_plot_':
        temp = updateCanvas(spacing, hor, ver)
    print(event, values)

window.Close()


# np.savetxt("lines.csv", lines, delimiter=",")


'''
for i in range(0,np.shape(lines)[0]):
    if [425] in lines[i]:
        if [340] in lines[i]:
            print(i)
'''

'''

# LSD line detector

lsd = cv2.createLineSegmentDetector(0)
lines1 = lsd.detect(np.uint8(smooth))[0]
angles0 = frac.getLineAngles(np.reshape(lines1, (np.shape(lines1)[0],4)))
image0 = frac.drawLines(np.reshape(lines1, (np.shape(lines1)[0],4)), angles,
                        np.shape(image))
frac.show_image(image0)
cv2.imwrite("lsd.png", image0)
'''


'''
dataset.transform * (3000,3000)

dataset.xy(3000,3000)

dataset.index(-37.62828781215599, -5.5309552901150445)

type(dataset)

import sys

sys.getsizeof(image)
'''
