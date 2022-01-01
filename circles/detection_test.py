import os
from pathlib import Path
from typing import List
import cv2
import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict


def get_program(number:int):
    return f"images\program-{number}.png"

def get_circles(img, debug, min_dist=50,max_radius = None, param2=50):
    if max_radius is None:
        max_radius = np.min(img.shape)

    circles = cv2.HoughCircles(
        image=img,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
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

def get_contour_centroid(cnt):
    M = cv2.moments(cnt)
    if M['m00'] ==0:
        return (0,0)
    return (int(M['m10']/M['m00']),int(M['m01']/M['m00']))

MAX_PROGRAM = 7
MIN_PROGRAM = 1
program = 5

image = cv2.imread(get_program(program))

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

hole_contour_indices = [n for l in list(hierarchy.values())[2:] for n in l]
hole_contours = np_area_contours[(hole_contour_indices,)]

print(list(hierarchy.values())[2:]+list(hierarchy.values())[0])
fill_contour_indices = [n for l in list(hierarchy.values())[:1]+list(hierarchy.values())[2:] for n in l]
fill_contours = np_area_contours[(fill_contour_indices,)]

gradient_moprh = morph(invert_gray, 2, cv2.MORPH_GRADIENT)

rects = [cv2.boundingRect(c) for c in fill_contours]
print(rects)
rect_kdtree = KDTree(rects)

connections = defaultdict(list)

holes = mask_contours(hole_contours, invert_gray)
for i, contour in enumerate(fill_contours):
    c = mask_contours([fill_contours[i]], invert_gray)

    
    not_c = cv2.bitwise_and(cv2.bitwise_not(c), invert_gray)

    dilated = morph_func(c, cv2.dilate, kernel_size=int(half_stroke_width*4))

    dilate_and_holes = cv2.bitwise_and(not_c, dilated)

    dil_erode = morph_func(dilate_and_holes, cv2.erode, 2)

    dil_erode_grad =morph( dil_erode, 2, cv2.MORPH_GRADIENT)

    grad_or_dil = cv2.bitwise_or(dil_erode_grad, gradient_moprh)

    filled_in = gradient_moprh.copy()

    connected_contour_intersects, _ = cv2.findContours(dil_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cci in connected_contour_intersects:
        cv2.floodFill(filled_in, None, (cci[0][0][0], cci[0][0][1]), 255)

    just_filled = filled_in-gradient_moprh
    just_filled_erode = morph_func(just_filled, cv2.erode, 2)

    connected_contours, _ = cv2.findContours(just_filled_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    c_rects = [cv2.boundingRect(cc) for cc in connected_contours]

    show = cv2.cvtColor(just_filled_erode, cv2.COLOR_GRAY2BGR)
    for r in c_rects:
        q = rect_kdtree.query(r)
        if q[0]<20:
            connections[i]+=[q[1]]

print(connections)

show_connections = image.copy()

for from_i in connections.keys():
    from_cnt = fill_contours[from_i]

    for to_i in connections[from_i]:
        to_cnt = fill_contours[to_i]

        cv2.line(show_connections, get_contour_centroid(from_cnt), get_contour_centroid(to_cnt), (255,0,0))

cv2.imshow("display", show_connections)
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
        elif chr(k)=='[':
            program-=1
        elif chr(k)==']':
            program+=1
        else:
            index=int(chr(k))
        program=np.clip(program, MIN_PROGRAM,MAX_PROGRAM)
        image = cv2.imread(get_program(program))
        
        index%=len(fill_contours)

        gray_blur = cv2.blur(invert_gray, (int(half_stroke_width),int(half_stroke_width)))

        show = image.copy()
        get_circles(gray_blur, show, min_dist=int(half_stroke_width), param2=65)
        from_cnt = fill_contours[index]

        for to_i in connections[index]:
            to_cnt = fill_contours[to_i]

            cv2.line(show, get_contour_centroid(from_cnt), get_contour_centroid(to_cnt), (255,0,0), thickness=2)

        cv2.imshow("display", show)
        
