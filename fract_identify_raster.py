#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 19:21:22 2019

@author: ademir
"""

import cv2
import numpy as np
import math

from PIL import Image

from skimage.segmentation import slic



import sys
sys.path.insert(0, '/home/ademir/Coding/Python/')

import rollinghough

from rollinghough import rht


import pywt #pywavelets

#For use in autoCanny function?
import argparse
import glob

def showImage (image):
    pil_img = Image.fromarray(image)
    pil_img.show()
    

def autoCanny(image, sigma=0.33):
    #https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged
   
    

################################################################################################
#Load image data
image = cv2.imread("DolinaXavier1e2_Orthomosaic_export_FriJan25121500f.png")
image = cv2.imread("DolinaXavier1e2_Orthomosaic_export_FriJan25121500f3.png")
image = cv2.imread("DolinaXAvierDenoisedhaarhardlevel4decomp10.tiff")
image = cv2.imread("DolinaXavier2_gray_Orthomosaic_export_FriJan25121500.tif")
image = cv2.imread("dolina2.tif")
image = cv2.imread("5.png")

#import matplotlib.pyplot as plt
#plt.imshow(image)
#plt.show()

#Resize the image
image = cv2.resize(image, (np.shape(image)[0]/2,np.shape(image)[1]/2), interpolation=0)
cv2.imwrite("resized.tif",image)
showImage(image)



#
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
showImage(gray)
eq =  cv2.equalizeHist(gray)
showImage(eq)

#Segmentação por cor
hsv =  cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_green = np.array([35,50,20])
upper_green = np.array([190,255,255])

mask = cv2.inRange(hsv, lower_green, upper_green)
mask = cv2.bitwise_not(mask)
res = cv2.bitwise_and(image, image, mask = mask)
showImage(res)

gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
showImage(gray)

#Segmentação por super pixel
segments = slic(image, n_segments=20, compactness=40)
segments = np.uint8(segments*10)
showImage(cv2.addWeighted(gray, 0.5, autoCanny(segments), 1, 0.0))
showImage(autoCanny(segments))




blur = cv2.GaussianBlur(gray,(5,5),1)
blur2 = cv2.GaussianBlur(gray,(5,5),2)
tophat = gray-blur


tophat = cv2.morphologyEx(eq, cv2.MORPH_TOPHAT, (5,5))
showImage(tophat)

thresholded = cv2.threshold(dilated, 100, 255, cv2.THRESH_BINARY)
showImage(thresholded[1])

dilated = cv2.dilate(tophat, (3,3), iterations = 2)
showImage(dilated)

openned = cv2.morphologyEx(thresholded[1], cv2.MORPH_OPEN, (3,3))
showImage(openned)

thresholded =  cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
showImage(thresholded[1])

rollinghough = rht(thresholded[1], 4, ntheta=180, background_percentile=25, verbose=True)


#Workin on gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
showImage(gray)


#padding
padded_image = []
#shape[0] = linha
#coluna[1] = coluna
shape = np.shape(gray)
add_line = int(pow(2, math.ceil(math.log(shape[0])/math.log(2)))- shape[0])
add_collumn = int(pow(2, math.ceil(math.log(shape[1])/math.log(2)))- shape[1])
padded_image = np.pad(gray, [(0, add_line),(0, add_collumn)], mode='constant', constant_values = 0)

cv2.imshow("Display", image)
################################################################################################
#wavelet decomposition
wavelet_decomp = pywt.wavedec2(gray, 'db4', level = 12)

levels = [-1,-4,-5,-6,-7,-8,-9,-10,-11,-12]
i = 0
for i in range(0,np.shape(levels)[0]):
    wavelet_decomp[levels[i]] = tuple([np.zeros_like(v) for v in wavelet_decomp[levels[i]]])

#wavelet_decomp[10] = pywt.threshold(wavelet_decomp[10], np.median(wavelet_decomp[10]), mode='soft', substitute=0)
#wavelet_decomp[11] = pywt.threshold(wavelet_decomp[11], np.percentile(wavelet_decomp[11], (60)), mode='less', substitute=0)


wavelet_recon = pywt.waverec2(wavelet_decomp,'db4')
showImage(wavelet_recon)
np.mean(wavelet_recon)
processed1 = np.uint8(cv2.normalize(wavelet_recon, None, 0, 255, cv2.NORM_MINMAX))
cv2.imwrite("wavelet.tif",processed1)

################################################################################################
#Threshold image
thresholded = pywt.threshold(gray, 80, mode='garotte', substitute=255)
showImage(thresholded)

thresholded =  cv2.threshold(processed1, 0.9*np.mean(processed1), 255, cv2.THRESH_BINARY)
showImage(thresholded[1])

################################################################################################
#Histogram equalization
processed2 = cv2.equalizeHist(processed1)
showImage(processed2)


################################################################################################
#Thinning borders
thinned = cv2.ximgproc.thinning(gray, thinningType = 0)
showImage(thinned)

################################################################################################
#denoise
gray_denoise = cv2.fastNlMeansDenoising(gray, None, 7, 21)
showImage(gray_denoise)

################################################################################################
#Smoothing image
#kernel = np.ones((5,5),np.float32)/25
#blur = cv2.filter2D(gray,-1,kernel)

blur = cv2.GaussianBlur(thresholded[1],(5,5),2)
showImage(blur)

################################################################################################
#Calculating threshold values for Canny edge detector
highThresh, threshIm = cv2.threshold(thresholded[1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
lowThresh = 0.5*highThresh

################################################################################################
#Canny edge detector
gray_canny = cv2.Canny(gray,lowThresh,highThresh)

gray_canny = cv2.Canny(thresholded[1],lowThresh,highThresh)

gray_canny = autoCanny(dilated)

gray_canny = cv2.Canny(blur,80,300)
cv2.imwrite("canny.tif",gray_canny)
showImage(gray_canny)

#gray_canny = cv2.ximgproc.thinning(gray_canny, thinningType = 0)


################################################################################################
#Generating the image with he lines
output = np.zeros(np.shape(gray_canny), np.uint8)

lines = []
rho = 1
theta = math.pi/180
minLineLength = 10
maxLineGap = 10


lines = cv2.HoughLinesP(gray_canny,rho,theta, threshold = 33, minLineLength = 5, maxLineGap = 2)


#angles = []
#i = 300
#for i in range(0,np.shape(lines)[0]):
#    angles.append(math.atan((lines[i][0][2]-lines[i][0][0])/(lines[i][0][3]-lines[i][0][1]))/(theta))


i = 0
for i in range(0,np.shape(lines)[0]):
    output = cv2.line(output, (lines[i][0][0],lines[i][0][1]), (lines[i][0][2],lines[i][0][3]), 1)
    
output = output*255
    
#showImage(output)
cv2.imwrite("hough.tif",output)



#laplacian = cv2.Laplacian(gray, 3)
#showImage(laplacian)

#sobel = cv2.Sobel(gray, 3, 3, 3)
#showImage(sobel)


################################################################################################
#Clean isolated points post Canny detection
filter = [0,0,0,0,255,0,0,0,0]
np.shape(gray_canny)
i = 0
j = 0
for i in range(0,np.shape(gray_canny)[0]):
    for j in range(0,np.shape(gray_canny)[1]):
        if ((i>0) and (i<np.shape(gray_canny)[0]-1) and (j>0) and (j<np.shape(gray_canny)[1]-1)):
            aux = [gray_canny[i-1][j-1],gray_canny[i-1][j],gray_canny[i-1][j+1],gray_canny[i][j-1],gray_canny[i][j],gray_canny[i][j+1],gray_canny[i+1][j-1],gray_canny[i+1][j],gray_canny[i+1][j+1]]
            if (np.array_equal(aux,filter)):
                print("hit")
                print("pixel[" + str(i) + "][" + str(j) + "]")
                gray_canny[i][j] = 0
                
                
pil_img = Image.fromarray(gray_canny)
#pil_img.show()
pil_img.save("canny_pil_clen1.tif")                