# -*- coding: utf-8 -*-
"""
Created on Sun Aug  10 02:21:49 2020

@author: adeju
"""

# import numpy as np
import PySimpleGUI as sg

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
                       enable_events=True, drag_submits=False, background_color='black',
                       graph_top_right=(500, 500), key="_Canvas1_")

vertical_slider = sg.Slider(
                        range=(100, 0), orientation='v', size=(20, 20),
                        enable_events=True, disable_number_display=True,
                        default_value=0, key="_Sl_vertical_")

horizontal_slider = sg.Slider(
                        range=(0, 100), orientation='h', size=(40, 20),
                        enable_events=True, disable_number_display=True,
                        default_value=0, key="_Sl_horizontal_")


image_processing = sg.Frame("Image processing", [
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
                     sg.Radio('Otsu', "RADIO3", disabled=False),
                     ],
                    [sg.Text('Skeletonization'),
                     sg.Radio('Lee', "RADIO4", default=True),
                     sg.Radio('Zhang', "RADIO4", disabled=True),
                     ],
                    [sg.Button('Image processing', key='_Bt_processing_',
                               disabled=True, visible=False,
                               size=(50, 1)),
                     ],
                    [sg.Text('Scale'),
                     sg.Slider(range=(1, 100), orientation='h', size=(10, 10),
                               default_value=100, key="_Sl_scale_")]
                    ], visible=True
                    )

hough_transform = sg.Frame("Hough transform", [
                    [sg.Text('Threshold'), sg.Slider(range=(1, 100),
                                                     orientation='h',
                                                     size=(10, 10),
                                                     default_value=20,
                                                     key="_Sl_threshold_"),
                     sg.Button('Hough', key='_Bt_hough_', disabled=True,
                               visible=False)],
                    [sg.Text('Min'),
                     sg.Slider(range=(2, 50), orientation='h', size=(10, 10),
                               default_value=2, key="_Sl_min_"),
                     sg.Text('Max'),
                     sg.Slider(range=(2, 50), orientation='h', size=(10, 10),
                               default_value=2, key="_Sl_max_")],
                         ])

line_connection = sg.Frame("Connecting lines", [
                 [sg.Text('Maximum length (px)'),
                  sg.Slider(range=(1, 100), orientation='h', size=(10, 10),
                            default_value=50, key="_Sl_radius_"),
                  # sg.InputCombo(('alpha', 'beta', 'distance', 'deviation'),
                  #               key='_Cb_mode_'),
                  sg.Button('Connect', key='_Bt_connect_', disabled=True,
                            visible=False)],
                 [sg.Text('alpha'), sg.Slider(range=(90, 180), orientation='h',
                                              size=(10, 10), default_value=134,
                                              key="_Sl_alpha_"),
                  sg.Text('beta'), sg.Slider(range=(90, 180), orientation='h',
                                             size=(10, 10), default_value=134,
                                             key="_Sl_beta_")]
                 ])

line_statistics = sg.Frame("Directional statistics", [
                 [sg.Text('Method'),
                  sg.InputCombo(('points', 'vertices'), key='_Cb_method_'),
                  sg.Text('Regression'), sg.InputCombo(('PCA', 'linear'),
                                                       key='_Cb_regression_'),
                  sg.Button('Rosechart', key='_Bt_rosechart_', disabled=True,
                            visible=False)]
                 ])

aerial_statistics = sg.Frame("Fracture statistics - intensity/spacing", [
                 [sg.Text('Box size'), sg.Input(key='_In_size_', size=(5, 5)),
                  sg.InputCombo(('pixels', 'meters'), key='_Cb_measure_'),
                  sg.InputCombo(('raster', 'svg'), key='_Cb_output_'),
                  sg.Button('Process', key='_Bt_fracture_statistics_',
                            disabled=True, visible=False)]
                 ])

layout = [[sg.Menu(menu_def, tearoff=True)],
          [sg.Button('Open', size=(10, 5)),
           sg.Button('Image processing', size=(10, 5), key='_Bt_processing_',
                     disabled=True),
           sg.Button('Line \ndetection', size=(10, 5), key='_Bt_hough_',
                     disabled=True),
           sg.Button('Line \nconnection', size=(10, 5), key='_Bt_connect_',
                     disabled=True),
           sg.Button('Line \nstatistics', size=(10, 5), key='_Bt_rosechart_',
                     disabled=True),
           sg.Button('Aerial \nstatistics', size=(10, 5),
                     key='_Bt_fracture_statistics_', disabled=True),
           sg.Button('Export', size=(10, 5)),
           ],
          [main_canvas, vertical_slider,
           sg.Frame("", [
                [image_processing, ],
                # [
                #     sg.Frame("Edge", [[sg.Button('Canny', key="_Bt_Canny_",
                #                                  disabled=True)]]), ],
                [hough_transform, ],
                [line_connection, ],
                # [line_statistics, ],
                [aerial_statistics, ],
                # [sg.Frame("Fracture statistics", [
                # [sg.Input(key='_In_point0_', size=(20,10)),
                # sg.Input(key='_In_point1_', size=(20,10)),
                # sg.Button('Scanline', key='_Bt_scanline_', disabled=True)],
                # [sg.Multiline(default_text="", key="_Mt_positions_"),
                # sg.Button('Draw points', key='_Bt_positions_')]
                # ])],

                ], element_justification='top')
           ],
          [horizontal_slider],
          load_buttons,
          [sg.Text('                                                                                   ', key="_Tx_position_", auto_size_text=True)]
          ]
