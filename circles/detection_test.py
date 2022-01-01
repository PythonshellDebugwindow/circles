import os
from pathlib import Path
from typing import List
import cv2
import numpy as np
from collections import defaultdict


def get_program(number:int):
    return f"images\program-{number}.png"

def get_circles(img, debug, max_radius = None, param2=50):
    if max_radius is None:
        max_radius = np.min(img.shape)

    circles = cv2.HoughCircles(
        image=img,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=300,
        param2=param2,
        maxRadius=max_radius,
    )

    if circles is not None:
        for c in circles[0]:
            cv2.circle(debug, (int(c[0]), int(c[1])), int(c[2]), (0, 0, 255), 3)
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

def mask_contours(cnts, img):
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, cnts, -1, (255, 255, 255), -1, cv2.LINE_AA)
    return mask

image = cv2.imread(get_program(5))

gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
gray = cv2.bitwise_not(gray)
gray = morph_func(gray, cv2.dilate)


debug_out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

circles = get_circles(gray, debug_out)

outer_contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
outer_contour_mask = mask_contours(outer_contours, gray)

gray_distance_transform = cv2.distanceTransform(gray, cv2.DIST_L2, 5)
half_stroke_width = np.max(gray_distance_transform)

outer_distance_transform = cv2.distanceTransform(outer_contour_mask, cv2.DIST_L2, 5)
line_width = np.unique(outer_distance_transform)

invert_gray = cv2.bitwise_not(gray)

def find_same_hierarchied(retr_tree, index):
    next_contours = []
    next_contour=retr_tree[index][0]

    while next_contour!=-1:
        next_contours.insert(0, next_contour)
        next_contour=retr_tree[next_contour][0]

    previous_contours = []
    previous_contour = retr_tree[index][1]

    while previous_contour!=-1:
        previous_contours.append(previous_contour)
        previous_contour=retr_tree[previous_contour][1]

    return previous_contours+[index]+next_contours

area_contours, (retr_hierarchy,) = cv2.findContours(invert_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

np_area_contours = np.array(area_contours)

first_found=None

for i,c in enumerate(retr_hierarchy):
    if c[3]==-1:
        first_found=i

hierarchy = {}

found_something = False
h_index=0
print(retr_hierarchy)
while retr_hierarchy[first_found][2]!=-1:
    same_h = find_same_hierarchied(retr_hierarchy, first_found)

    hierarchy[h_index]=same_h

    for c in same_h:
        if retr_hierarchy[c][2]!=-1:
            first_found=retr_hierarchy[c][2]
            h_index+=1
            found_something=True

hierarchy[h_index]=find_same_hierarchied(retr_hierarchy, first_found)

print(hierarchy)

print(half_stroke_width)

cv2.imshow("display", invert_gray)

hole_contour_indices = [n for l in list(hierarchy.values())[2:] for n in l]
hole_contours = np_area_contours[(hole_contour_indices,)]

gradient_moprh = morph(invert_gray, 2, cv2.MORPH_GRADIENT)

index=0
while True:
    k=cv2.waitKey(0)

    if chr(k)=="q":
        exit()
    
    else:
        if chr(k)=='.':
            index+=1
        elif chr(k)==',':
            index-=1
        else:
            index=int(chr(k))
        index%=len(area_contours)

        c = mask_contours([area_contours[index]], invert_gray)

        holes = mask_contours(hole_contours, invert_gray)
        
        not_c = cv2.bitwise_and(cv2.bitwise_not(c), invert_gray)

        dilated = morph_func(c, cv2.dilate, kernel_size=int(half_stroke_width*4))

        # index%=len(hierarchy)
        # print((tuple(hierarchy[index]),))


        dilate_and_holes = cv2.bitwise_and(not_c, dilated)

        dil_erode_grad =morph( morph_func(dilate_and_holes, cv2.erode, 4), 2, cv2.MORPH_GRADIENT)

        grad_or_dil = cv2.bitwise_or(dil_erode_grad, gradient_moprh)

        cv2.imshow("display", grad_or_dil)
        
