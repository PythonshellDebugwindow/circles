import os
from pathlib import Path
import cv2
import numpy as np

def get_program(number:int):
    return f"images\program-{number}.png"

def get_circles(img, debug):
        circles = cv2.HoughCircles(
            image=img,
            method=cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=300,
            param2=53,
            maxRadius=0,
        )

        if circles is not None:
            for c in circles[0]:
                cv2.circle(debug, (int(c[0]), int(c[1])), int(c[2]), (0, 0, 255), 1)
                cv2.circle(debug, (int(c[0]), int(c[1])), 2, (0, 255, 0), -1)
                pass

        return circles

def get_lines(img, debug):
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,180)
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(debug,(x1,y1),(x2,y2),(0,0,255),2)

def morph(img, kernel_size=2, morph=cv2.MORPH_BLACKHAT):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    return cv2.morphologyEx(img, morph, kernel)

def morph_func(img, func, kernel_size=2):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    return func(img, kernel)

image = cv2.imread(get_program(2))

gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
gray = cv2.bitwise_not(gray)
gray = morph_func(gray, cv2.dilate)

debug_out = image.copy()
# debug_out = cv2.bitwise_not(debug_out)
# debug_out = morph(debug_out, kernel_size=2, morph=cv2.MORPH_GRADIENT)

get_circles(gray, debug_out)

# get_lines(gray, debug_out)

cv2.imshow("display", debug_out)
k=cv2.waitKey(0)