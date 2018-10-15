import cv2
import numpy as np
import argparse

import math
from scipy.spatial import distance


#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required = True, help = "Path to the image")
#args = vars(ap.parse_args())

#img = cv2.imread(args["image"])
img = cv2.imread("test.jpg", 0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

cv2.namedWindow("Display", flags= cv2.WINDOW_AUTOSIZE)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,100,
                           param1=50,param2=25,minRadius=10,maxRadius=30)

circles = np.uint16(np.around(circles))

#coordinates of pupils
point1x = 0.0
point1y = 0.0
point2x = 0.0
point2y = 0.0

#coordinates of object on forehead
leftpointx = 0
leftpointy = 0
rightpointx = 0
rightpointy = 0

#coordinates of pupils in case hough circles doesn't work
leftpointx2 = 0
leftpointy2 = 0
rightpointx2 = 0
rightpointy2 = 0

objectexist = 0

m = 0
k = 0

dist = 0 #distance of pupils in pixels
dist2 = 0 #distance of object in pixels

mm = 100 #distance of object in mm



n = 1


for i in circles[0,:]:
    # draw the outer circle

    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    if n == 1:
        n = 2
        point1x = float(i[0])
        point1y = float(i[1])
    elif n==2 :
        point2x = float(i[0])
        point2y = float(i[1])

    #print(" " + point1x + ", " + point1y + ", " + point2x + ", " + point2y)

print(point1x)
print(point1y)
print(point2x)
print(point2y)





dist = math.hypot(point2x - point1x, point2y - point1y) #ipd in pixels

#if dist < 10000:
print(dist)



def my_mouse_callback(event, x, y, flags, param):
    global m
    global k
    global leftpointx
    global leftpointy
    global rightpointx
    global rightpointy

    global leftpointx2
    global leftpointy2
    global rightpointx2
    global rightpointy2

    global objectexist

    global dist
    global dist2
    if event==cv2.EVENT_LBUTTONDBLCLK:
        m = m+1

        #print(x)
        if m == 1:
            leftpointx = x
            leftpointy = y
        elif m == 2:
            rightpointx = x
            rightpointy = y
            dist2 = math.hypot(rightpointx - leftpointx, rightpointy - leftpointy)
            print( "object distance = " + str(dist2))

            if dist < 10000:
                distanceInMM = dist * (mm / dist2)  # ipd in mm
                print( "IPD in mm = " + str(distanceInMM))
    if dist > 10000:
        if event==cv2.EVENT_RBUTTONDBLCLK: #used if hough circles doesn't work
            k = k+1
            if dist2 == 0:
                dist2 == 500

            #print(x)
            if k == 1:
                leftpointx2 = x
                leftpointy2 = y
            elif k == 2:
                rightpointx2 = x
                rightpointy2 = y
                dist = math.hypot(rightpointx2 - leftpointx2, rightpointy2 - leftpointy2)
                print("IPD in pixels = " + str(dist))

                distanceInMM = dist * (mm / dist2)  # ipd in mm
                print( "IPD in mm = " + str(distanceInMM))

cv2.setMouseCallback("Display",my_mouse_callback,cimg)










while(1):
    cv2.imshow("Display",cimg)

    if dist2 == 0 and dist < 10000:
        c = cv2.waitKey(0)
        if c == 32: #spacebar
            dist2 = 500
            distanceInMM = dist * (mm / dist2)  # ipd in mm
            print("IPD in mm = " + str(distanceInMM))

    if cv2.waitKey(15)%0x100==27:break    # waiting for clicking escape key
cv2.destroyWindow("Display")
